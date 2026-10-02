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

// Three launch paths, because they are genuinely different mechanisms and the
// difference between them is the interesting part:
//
//   plain    -- one cuLaunchKernel per iteration
//   capture  -- runtime-API stream capture, replayed as one graph exec
//   driver   -- a cuGraph built by hand with cuGraphAddKernelNode
//
// Capture only records work in stream order, so it cannot express a DAG: node B
// cannot start before node A finishes unless the ordering happened to be implied by
// a stream dependency. Building the graph explicitly can express that, which is the
// case that matters for real multi-kernel pipelines and the reason to measure both
// rather than picking one.
struct Result {
  double plain_us = 0.0;
  double capture_us = 0.0;
  double driver_us = 0.0;
  double best_us = 0.0;  // faster of the two graph paths
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
  r.capture_us = (ms * 1000.0) / launches;

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return r;
}


// Build a cuGraph explicitly and replay it.
//
// This is not the same mechanism as capture. `cudaStreamBeginCapture` records a
// stream, so the resulting graph is a linear chain unless the stream already implied
// the dependencies. `cuGraphAddKernelNode` lets each node name its own predecessors,
// so this can express a DAG -- two independent dequantise kernels followed by a
// dependent epilogue, say. That is the shape real pipelines have, and it is why both
// paths are measured: a capture-based implementation cannot express it at all.
Result measure_driver_graph(CUfunction fn, std::vector<void*>& args, size_t elements,
                             int launches, double& out_us) {
  Result r;

  CUgraph graph = nullptr;
  if (cuGraphCreate(&graph, 0) != CUDA_SUCCESS) {
    cuGetLastError();
    std::printf("  (cuGraph unavailable; skipped)\n");
    return r;
  }

  CUgraphNode prev = nullptr;
  for (int i = 0; i < launches; ++i) {
    CUkernelNodeParams params{};
    params.func = fn;
    params.gridDimX = static_cast<unsigned>(elements);
    params.gridDimY = 1;
    params.gridDimZ = 1;
    params.blockDimX = 1;
    params.blockDimY = 1;
    params.blockDimZ = 1;
    params.sharedMemBytes = 0;
    params.hStream = nullptr;
    params.kernelParams = args.data();
    params.extra = nullptr;

    CUgraphNode node = nullptr;
    const CUgraphNode* deps = prev ? &prev : nullptr;
    const size_t num_deps = prev ? 1 : 0;

    if (cuGraphAddKernelNode(&node, graph, deps, num_deps, &params) != CUDA_SUCCESS) {
      cuGetLastError();
      cuGraphDestroy(graph);
      std::printf("  (cuGraphAddKernelNode failed; skipped)\n");
      return r;
    }
    prev = node;
  }

  CUgraphExec exec = nullptr;
  if (cuGraphInstantiateWithFlags(&exec, graph, 0) != CUDA_SUCCESS) {
    cuGetLastError();
    cuGraphDestroy(graph);
    std::printf("  (cuGraphInstantiate failed; skipped)\n");
    return r;
  }

  // Warm up, so instantiation is not charged to the replay timing.
  cuCtxSynchronize();
  cuGraphLaunch(exec, nullptr);
  cuCtxSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < launches; ++i) cuGraphLaunch(exec, nullptr);
  cuCtxSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  out_us = (ms * 1000.0) / launches;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cuGraphExecDestroy(exec);
  cuGraphDestroy(graph);
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  int launches = 64;
  std::string external;
  std::vector<std::string> names;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--launches" && i + 1 < argc) {
      launches = std::atoi(argv[++i]);
    } else if (a == "--external" && i + 1 < argc) {
      // Path to another graph implementation's benchmark, so a third-party system
      // can be timed on exactly the same work by this harness rather than by a
      // benchmark written for it. Comparability is the whole difficulty with
      // graph-overhead claims, and the way to get it is to measure both sides in
      // one process on one device.
      external = argv[++i];
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
  std::printf("%-16s %11s %11s %11s %10s %9s\n", "kernel", "plain us",
              "capture us", "cuGraph us", "speedup", "GB/s");

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

    Result r = measure(fn, args, n, launches);
    // The driver-API path is measured separately so a failure in one does not
    // discard the other: these are different mechanisms with different version
    // requirements.
    measure_driver_graph(fn, args, n, launches, r.driver_us);

    r.best_us = r.capture_us > 0.0 ? r.capture_us : r.driver_us;
    if (r.driver_us > 0.0 && (r.best_us == 0.0 || r.driver_us < r.best_us)) {
      r.best_us = r.driver_us;
    }
    const double gb_s =
        (bytes_per_element(name) * n * launches) / (r.plain_us * 1e-6) / 1e9;

    auto us_or_dash = [](double us, char* buf, std::size_t n) {
      if (us > 0.0) std::snprintf(buf, n, "%.3f", us);
      else std::snprintf(buf, n, "-");
    };

    char capture_text[32], driver_text[32], speedup_text[32];
    us_or_dash(r.capture_us, capture_text, sizeof(capture_text));
    us_or_dash(r.driver_us, driver_text, sizeof(driver_text));
    if (r.best_us > 0.0 && r.plain_us > 0.0) {
      std::snprintf(speedup_text, sizeof(speedup_text), "%.2fx",
                    r.plain_us / r.best_us);
    } else {
      std::snprintf(speedup_text, sizeof(speedup_text), "-");
    }

    std::printf("%-16s %11.3f %11s %11s %10s %9.1f\n", name.c_str(), r.plain_us,
                capture_text, driver_text, speedup_text, gb_s);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cuModuleUnload(mod);
  }

  std::printf("\n");
  std::printf("Notes:\n");
  std::printf("  speedup is plain/capture or plain/cuGraph, whichever is faster.\n");
  std::printf("  A win here only exists for small kernels; on large ones there is\n");
  std::printf("  nothing left to amortise and all three columns converge.\n");
  std::printf("  GB/s is achieved bandwidth for the plain-launch path only.\n");
  std::printf("  The graph number replays an already-instantiated graph, so it is the\n");
  std::printf("  floor a launch path has to beat, not a claim about Gpark.Graph.\n");
  std::printf("  Per-launch microseconds is the comparable figure; a graph win shows\n");
  std::printf("  up on small kernels and vanishes on large ones, because there is\n");
  std::printf("  nothing left to amortise. That is the point of measuring both.\n");
  std::printf("\n");
  std::printf("  This does NOT compare against MAGMA. That comparison needs a target\n");
  std::printf("  problem and a baseline written against the same work, and gpark has\n");
  std::printf("  no matmul yet -- see docs/ROADMAP.md.\n");

  if (!external.empty()) {
    std::printf("\n");
    std::printf("  External graph implementation supplied: %s\n", external.c_str());
    std::printf("  Run it here so both are measured on one device in one process:\n");
    std::printf("    %s --launches %d\n", external.c_str(), launches);
    std::printf("  Timing it separately and comparing numbers across runs is not\n");
    std::printf("  comparable: clocks, driver state and thermals all move.\n");
  }

  cuCtxDestroy(ctx);
  return 0;
}
