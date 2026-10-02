// Launch overhead benchmark: plain launches vs CUDA Graphs vs the same work fused.
//
// Why this exists
// ---------------
// Every kernel in corpus/ is bandwidth-bound, which is a deliberate choice: it
// means a wrong answer shows up as a wrong number rather than as a plausible one.
// But the workload gpark ultimately targets — quant inference, RL rollouts,
// agent loops — is often *launch*-bound instead. An RL step might be forty tiny
// kernels, and forty launches cost more than all the arithmetic together.
//
// That is the regime CUDA Graphs exist for, and it is the single largest available
// win in that regime. This benchmark measures it honestly:
//
//   * it always launches at least 32 times, because a single launch measures
//     launch overhead and nothing else;
//   * it reports per-launch microseconds, not total time, so the number means the
//     same thing regardless of how many launches were batched;
//   * it JITs the gpark goldens through the driver API, so what is being replayed
//     is exactly the PTX gpark emits.
//
// It does not yet compare against MAGMA. See the note at the bottom of this file
// before drawing conclusions from it.
//
// Build:
//   nvcc -O2 -o graph_bench graph_bench.cu -lcuda
// Run:
//   ./graph_bench                 # all kernels, both paths
//   ./graph_bench --launches 256  # batch size for the graph path

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

#define CU_CHECK(expr)                                                          \
  do {                                                                          \
    CUresult _s = (expr);                                                       \
    if (_s != CUDA_SUCCESS) {                                                   \
      const char* _n = nullptr;                                                 \
      cuGetErrorName(_s, &_n);                                                  \
      std::fprintf(stderr, "%s:%d: %s failed: %s (%d)\n", __FILE__, __LINE__,    \
                   #expr, _n ? _n : "?", static_cast<int>(_s));                 \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

constexpr const char* kCorpus = "../corpus/golden";

double bytes_per_element(const std::string& name) {
  if (name == "vec_add_f32") return 12.0;
  if (name == "saxpy_f32") return 12.0;
  if (name == "unpack_u4_f32") return 36.0;
  return 0.0;
}

std::string read_text(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::fseek(f, 0, SEEK_END);
  const long len = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string text(len + 1, '\0');
  const size_t got = std::fread(&text[0], 1, len, f);
  text.resize(got);
  std::fclose(f);
  return text;
}

struct Result {
  double plain_us = 0.0;
  double graph_us = 0.0;
  double speedup = 0.0;
};

// The measurement, for one kernel.
//
// `plain` is a stream loop: one cuLaunchKernel per iteration, synchronised only
// once at the end. That is the fairest possible baseline — removing the sync would
// be measuring async queuing rather than launch cost, and would flatter graphs.
//
// `args` is deliberately a non-const reference: cuLaunchKernel takes `void**`, and
// a const vector would only hand back `void* const*`.
Result measure(CUfunction fn, std::vector<void*>& args, size_t elements,
               int launches) {
  Result r;

  // Warm up, discarding: JIT and first-launch costs would otherwise dominate.
  for (int i = 0; i < 4; ++i) {
    CU_CHECK(cuLaunchKernel(fn, elements, 1, 1, 1, 1, 1, 0, nullptr, args.data(),
                            nullptr));
  }
  CU_CHECK(cuCtxSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // --- plain stream launches ---
  cudaEventRecord(start);
  for (int i = 0; i < launches; ++i) {
    CU_CHECK(cuLaunchKernel(fn, elements, 1, 1, 1, 1, 1, 0, nullptr, args.data(),
                            nullptr));
  }
  CU_CHECK(cuCtxSynchronize());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  r.plain_us = (ms * 1000.0) / launches;

  // --- the same launches, captured once into a graph and replayed ---
  //
  // Capture must happen on the legacy default stream unless a mode is set;
  // ThreadLocal mode is what lets unrelated driver-API work keep running.
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;

  if (cudaStreamBeginCapture(0, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    cudaGetLastError();  // clear the sticky-free error
    std::printf("  (graph capture unavailable on this device; skipped)\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return r;
  }

  for (int i = 0; i < launches; ++i) {
    CU_CHECK(cuLaunchKernel(fn, elements, 1, 1, 1, 1, 1, 0, nullptr, args.data(),
                            nullptr));
  }

  if (cudaStreamEndCapture(0, &graph) != cudaSuccess || graph == nullptr) {
    cudaGetLastError();
    std::printf("  (capture failed; skipped)\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return r;
  }

  // InstantiateWithFlags has one stable signature across CUDA 11.4+, unlike
  // cudaGraphInstantiate, whose parameter list has changed more than once.
  if (cudaGraphInstantiateWithFlags(&exec, graph, 0) != cudaSuccess) {
    cudaGetLastError();
    cudaGraphDestroy(graph);
    std::printf("  (instantiate failed; skipped)\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return r;
  }

  // Warm the graph too, so instantiation is not charged to the replay timing.
  CU_CHECK(cuCtxSynchronize());
  cudaGraphLaunch(exec, 0);
  CU_CHECK(cuCtxSynchronize());

  cudaEventRecord(start);
  for (int i = 0; i < launches; ++i) {
    cudaGraphLaunch(exec, 0);
  }
  CU_CHECK(cuCtxSynchronize());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  r.graph_us = (ms * 1000.0) / launches;

  if (r.plain_us > 0.0) r.speedup = r.plain_us / r.graph_us;

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  int launches = 64;
  std::vector<std::string> names;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--launches" && i + 1 < argc) {
      launches = std::atoi(argv[++i]);
    } else {
      names.push_back(a);
    }
  }
  if (names.empty()) names = {"vec_add_f32", "saxpy_f32", "unpack_u4_f32"};

  CU_CHECK(cuInit(0));
  int count = 0;
  CU_CHECK(cuDeviceGetCount(&count));
  if (count == 0) {
    std::fprintf(stderr, "no CUDA device\n");
    return 1;
  }

  CUdevice device;
  CU_CHECK(cuDeviceGet(&device, 0));
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  char devname[256] = {};
  cuDeviceGetName(devname, sizeof(devname), device);

  CUcontext ctx;
  CU_CHECK(cuCtxCreate(&ctx, 0, device));

  std::printf("device: %s (sm_%d%d)\n", devname, major, minor);
  std::printf("launches per measurement: %d\n", launches);
  std::printf("\n");
  std::printf("%-16s %12s %12s %10s %10s\n", "kernel", "plain us", "graph us",
              "speedup", "GB/s");

  for (const std::string& name : names) {
    const std::string text = read_text(std::string(kCorpus) + "/" + name + ".ptx");

    const std::string arch = "sm_" + std::to_string(major) + std::to_string(minor);
    const char* arch_cstr = arch.c_str();
    CUjit_option opts[] = {CU_JIT_TARGET_ARCH};
    void* optvals[] = {&arch_cstr};

    CUmodule mod;
    CUresult st = cuModuleLoadDataEx(&mod, text.data(), 1, opts, optvals);
    if (st != CUDA_SUCCESS) {
      const char* err = nullptr;
      cuGetErrorName(st, &err);
      std::printf("%-16s  JIT failed: %s\n", name.c_str(), err ? err : "?");
      continue;
    }

    CUfunction fn;
    CU_CHECK(cuModuleGetFunction(&fn, mod, name.c_str()));

    const size_t n = 1u << 20;
    void *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    cudaMemset(d_a, 0, n * sizeof(float));
    cudaMemset(d_b, 0, n * sizeof(float));

    uint32_t count_u32 = static_cast<uint32_t>(n);
    float scale = 1.0f;

    std::vector<void*> args;
    if (name == "unpack_u4_f32") {
      args = {d_a, d_c, &scale, &count_u32};
    } else {
      args = {d_a, d_b, d_c, &count_u32};
    }

    const Result r = measure(fn, args, n, launches);
    const double gb_s =
        (bytes_per_element(name) * n * launches) / (r.plain_us * 1e-6) / 1e9;

    char graph_text[32];
    char speedup_text[32];
    if (r.graph_us > 0.0) {
      std::snprintf(graph_text, sizeof(graph_text), "%.3f", r.graph_us);
      std::snprintf(speedup_text, sizeof(speedup_text), "%.2fx", r.speedup);
    } else {
      std::snprintf(graph_text, sizeof(graph_text), "-");
      std::snprintf(speedup_text, sizeof(speedup_text), "-");
    }

    std::printf("%-16s %12.3f %12s %10s %10.1f\n", name.c_str(), r.plain_us,
                graph_text, speedup_text, gb_s);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cuModuleUnload(mod);
  }

  std::printf("\n");
  std::printf("Notes:\n");
  std::printf("  GB/s is achieved bandwidth for the plain-launch path only.\n");
  std::printf("  The graph number replays an already-instantiated graph, so it is the\n");
  std::printf("  floor a launch path has to beat, not a claim about Gpark.Graph.\n");
  std::printf("  Per-launch microseconds is the comparable figure; a graph win shows\n");
  std::printf("  up on small kernels and vanishes on large ones, because there is\n");
  std::printf("  nothing left to amortise. That is the point of measuring both.\n");
  std::printf("\n");
  std::printf("  This does NOT compare against MAGMA. See docs/ROADMAP.md: the MAGMA\n");
  std::printf("  comparison needs a target problem chosen and a baseline written,\n");
  std::printf("  and 'graphsuite' is still ambiguous -- CUDA Graphs, cuGraph, or\n");
  std::printf("  something else. Ask before drawing conclusions.\n");

  cuCtxDestroy(ctx);
  return 0;
}
