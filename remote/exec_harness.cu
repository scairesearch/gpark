// Executes the corpus goldens on a real GPU and checks them against CPU
// references.
//
// Deliberately built against the CUDA *driver* API, not the runtime API, and
// deliberately loads .ptx files at run time rather than linking generated fatbins.
// Two reasons:
//
//   1. cuModuleLoadData JITs PTX directly, so gpark's output reaches the hardware
//      without ever passing through nvcc. That is the whole claim being tested: the
//      PTX gpark emits *is* the program. An nvcc round trip in the middle would
//      hide exactly the class of bug worth finding.
//
//   2. Nothing to rebuild when a golden changes. `make corpus && make remote` is
//      the whole loop, which is what makes running this on every edit practical.
//
// Build:
//   nvcc -O2 -o exec_harness exec_harness.cu -lcuda
// Run:
//   ./exec_harness                  # every kernel it knows a reference for
//   ./exec_harness vec_add_f32      # one
//   ./exec_harness --repeat 1000    # timing runs
//   ./exec_harness --arch sm_90
//
// See docs/VALIDATION.md for the tolerance rationale. In short: exact equality for
// integer kernels, tolerance for float, and never a tolerance wide enough to hide a
// genuinely wrong kernel.

#include <cuda.h>
// Included explicitly rather than relying on nvcc to pull in the runtime: the
// driver API does not include it, and a source file that only compiles under nvcc
// cannot be syntax-checked by anything else.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr const char* kCorpus = "../corpus/golden";

// ---------------------------------------------------------------------------
// Driver API error handling
// ---------------------------------------------------------------------------

#define CU_CHECK(expr)                                                          \
  do {                                                                          \
    CUresult _status = (expr);                                                  \
    if (_status != CUDA_SUCCESS) {                                              \
      const char* _name = nullptr;                                              \
      cuGetErrorName(_status, &_name);                                          \
      std::fprintf(stderr, "%s:%d: %s failed: %s (%d)\n", __FILE__, __LINE__,   \
                   #expr, _name ? _name : "?", static_cast<int>(_status));      \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t _status = (expr);                                               \
    if (_status != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr, \
                   cudaGetErrorString(_status));                                \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// ---------------------------------------------------------------------------
// CPU references
//
// Written twice on purpose: once naively, once in the exact order the kernel uses.
// A single reference would hide summation-order differences, and those are exactly
// what a reduction kernel gets wrong.
// ---------------------------------------------------------------------------

void ref_vec_add(const std::vector<float>& a, const std::vector<float>& b,
                 std::vector<float>& out) {
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
}

void ref_saxpy(float alpha, const std::vector<float>& x,
               const std::vector<float>& y, std::vector<float>& out) {
  for (size_t i = 0; i < x.size(); ++i) out[i] = alpha * x[i] + y[i];
}

// Butterfly reduction sums pairs at decreasing strides. The reference has to match
// that order, because in a different order the result differs in the last bits and
// a bitwise comparison would be measuring the wrong thing.
float ref_reduce_sum_butterfly(const std::vector<float>& in) {
  std::vector<float> v = in;
  for (size_t stride = 16; stride >= 1; stride >>= 1) {
    for (size_t lane = 0; lane < v.size(); ++lane) {
      const size_t partner = lane ^ stride;
      if (partner > lane && partner < v.size()) {
        // Preserve the pair's *first* operand as the accumulator, matching
        // shfl.sync.bfly + add, where each lane keeps its own value.
        v[lane] = v[lane] + v[partner];
        v[partner] = v[lane];
      }
    }
  }
  return v[0];
}

float ref_reduce_sum(const std::vector<float>& in) {
  float acc = 0.0f;
  for (float x : in) acc += x;
  return acc;
}

std::vector<float> ref_unpack_u4(const std::vector<uint32_t>& packed, float scale) {
  std::vector<float> out(packed.size() * 8);
  for (size_t w = 0; w < packed.size(); ++w) {
    for (int k = 0; k < 8; ++k) {
      out[w * 8 + k] = static_cast<float>((packed[w] >> (4 * k)) & 0xF) * scale;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

struct Diff {
  double max_abs = 0.0;
  double max_rel = 0.0;
  size_t worst = 0;
  size_t mismatched = 0;
};

template <typename T>
Diff compare(const std::vector<T>& got, const std::vector<T>& want, double tol) {
  Diff d;
  for (size_t i = 0; i < want.size(); ++i) {
    if (got[i] == want[i]) continue;

    const double a = static_cast<double>(got[i]);
    const double b = static_cast<double>(want[i]);
    const double abs_err = std::fabs(a - b);
    const double rel_err = abs_err / (std::fabs(b) > 1e-12 ? std::fabs(b) : 1.0);

    ++d.mismatched;
    if (abs_err > d.max_abs) {
      d.max_abs = abs_err;
      d.worst = i;
    }
    d.max_rel = std::max(d.max_rel, rel_err);
  }
  (void)tol;
  return d;
}

// ---------------------------------------------------------------------------
// Device buffer
// ---------------------------------------------------------------------------

template <typename T>
class DeviceArray {
 public:
  explicit DeviceArray(size_t count) : count_(count) {
    if (count_ > 0) CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
  }
  ~DeviceArray() {
    if (ptr_) cudaFree(ptr_);
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  void upload(const std::vector<T>& host) {
    CUDA_CHECK(cudaMemcpy(ptr_, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
  }
  void download(std::vector<T>& host) const {
    host.resize(count_);
    CUDA_CHECK(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }
  T* get() const { return static_cast<T*>(ptr_); }

 private:
  void* ptr_ = nullptr;
  size_t count_;
};

// ---------------------------------------------------------------------------
// Timing
//
// A correctness pass tells you whether a kernel works. It tells you nothing about
// whether it is fast, which is the only question gpark exists to answer. So every
// kernel reports achieved bandwidth, not just elapsed time: on the kernels here
// arithmetic intensity is roughly zero, and bytes/second is the number that either
// approaches roofline or exposes that it will not.
// ---------------------------------------------------------------------------

// Bytes each kernel moves per element, for the achieved-bandwidth calculation.
double bytes_per_element(const std::string& name) {
  if (name == "vec_add_f32") return 12.0;   // 2 loads + 1 store, 4 bytes each
  if (name == "saxpy_f32") return 12.0;
  if (name == "unpack_u4_f32") return 36.0; // 4 in + 32 out
  return 0.0;
}

void time_it(CUfunction fn, void** kernel_args, int repeat, size_t elements,
             const char* name = "") {
  if (repeat <= 0) return;

  // Warm up: the first launch pays for JIT, module load and context setup, and
  // averaging that into the result makes every number meaningless.
  CU_CHECK(cuLaunchKernel(fn, elements, 1, 1, 1, 1, 1, 0, nullptr, kernel_args, nullptr));
  CU_CHECK(cuCtxSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  for (int i = 0; i < repeat; ++i) {
    CU_CHECK(cuLaunchKernel(fn, elements, 1, 1, 1, 1, 1, 0, nullptr, kernel_args,
                            nullptr));
  }
  CUDA_CHECK(cuCtxSynchronize());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  const double per_launch_us = (ms * 1000.0) / repeat;
  const double bpe = bytes_per_element(name[0] ? name : "");
  if (bpe > 0.0) {
    const double gb_s = (bpe * elements * repeat) / (ms * 1e-3) / 1e9;
    std::printf("               %8.3f us/launch  %6.1f GB/s\n", per_launch_us, gb_s);
  } else {
    std::printf("               %8.3f us/launch\n", per_launch_us);
  }
}

// ---------------------------------------------------------------------------
// Kernel runner
// ---------------------------------------------------------------------------

struct Args {
  std::vector<std::string> names;
  int repeat = 0;
  std::string arch = "sm_80";
};

bool run_vec_add(CUmodule module, int repeat, uint32_t n) {
  CUfunction fn;
  CU_CHECK(cuModuleGetFunction(&fn, module, "vec_add_f32"));

  std::vector<float> a(n), b(n), want(n), got(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(i) * -1.25f + 3.0f;
  }
  ref_vec_add(a, b, want);

  DeviceArray<float> da(n), db(n), dout(n);
  da.upload(a);
  db.upload(b);

  void* args[] = {da.get(), db.get(), dout.get(), &n};
  CU_CHECK(cuLaunchKernel(fn, n, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  CU_CHECK(cuCtxSynchronize());

  dout.download(got);
  const Diff d = compare(got, want, 0.0);
  if (d.mismatched != 0) {
    std::printf("  vec_add_f32  FAIL %zu/%zu differ, max_abs=%g at %zu\n",
                d.mismatched, want.size(), d.max_abs, d.worst);
    return false;
  }
  std::printf("  vec_add_f32  ok   %u elements\n", n);
  time_it(fn, args, repeat, n, "vec_add_f32");
  return true;
}

bool run_saxpy(CUmodule module, int repeat, uint32_t n) {
  CUfunction fn;
  CU_CHECK(cuModuleGetFunction(&fn, module, "saxpy_f32"));

  // Not const: cuLaunchKernel takes void**, and a pointer to a const T does not
  // convert to void*. The kernel only reads these.
  float alpha = 0.75f;
  std::vector<float> x(n), y(n), want(n), got(n);
  for (uint32_t i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i % 97) * 0.03125f;
    y[i] = static_cast<float>(i % 31) - 15.0f;
  }
  ref_saxpy(alpha, x, y, want);

  DeviceArray<float> dx(n), dy(n), dout(n);
  dx.upload(x);
  dy.upload(y);

  void* args[] = {dx.get(), dy.get(), dout.get(), &alpha, &n};
  CU_CHECK(cuLaunchKernel(fn, n, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  CU_CHECK(cuCtxSynchronize());

  dout.download(got);
  const Diff d = compare(got, want, 1e-6);
  if (d.mismatched != 0) {
    std::printf("  saxpy_f32    FAIL %zu/%zu differ, max_abs=%g at %zu\n",
                d.mismatched, want.size(), d.max_abs, d.worst);
    return false;
  }
  std::printf("  saxpy_f32    ok   %u elements, alpha=%g\n", n, alpha);
  time_it(fn, args, repeat, n, "saxpy_f32");
  return true;
}

bool run_reduce(CUmodule module, int repeat, uint32_t n) {
  CUfunction fn;
  CU_CHECK(cuModuleGetFunction(&fn, module, "reduce_sum_f32"));

  // n must be a warp multiple: the kernel clamps out-of-range lanes, which keeps
  // the shuffle tree well-formed but duplicates values. This is documented in the
  // kernel, and here it is enforced rather than discovered.
  uint32_t padded = (n + 31) & ~31u;  // not const: passed by address to cuLaunchKernel
  std::vector<float> in(padded, 0.0f), got(1);
  for (uint32_t i = 0; i < padded; ++i) in[i] = static_cast<float>(i % 13) - 6.0f;

  DeviceArray<float> din(padded), dout(1);
  din.upload(in);
  void* args[] = {din.get(), dout.get(), &padded};
  CU_CHECK(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  CU_CHECK(cuCtxSynchronize());

  dout.download(got);
  const float want_butterfly = ref_reduce_sum_butterfly(in);
  const float want_sequential = ref_reduce_sum(in);

  // Compare against the sequential reference, but with a tolerance derived from
  // the gap between the two reference orderings. That gap is a property of
  // parallel reduction, not of gpark: demanding exact equality with a sequential
  // sum would be demanding something no parallel reduction can deliver.
  const float tol = std::max(1e-4f, std::fabs(want_sequential - want_butterfly));
  const float err = std::fabs(got[0] - want_sequential);
  if (!(err <= tol) || std::isnan(got[0])) {
    std::printf("  reduce_sum   FAIL got=%g want=%g err=%g tol=%g\n", got[0],
                want_sequential, err, tol);
    return false;
  }
  std::printf("  reduce_sum   ok   sum=%g (sequential ref %g, tol %g)\n", got[0],
              want_sequential, tol);
  time_it(fn, args, repeat, padded, "reduce_sum_f32");
  return true;
}

bool run_unpack(CUmodule module, int repeat, uint32_t words) {
  CUfunction fn;
  CU_CHECK(cuModuleGetFunction(&fn, module, "unpack_u4_f32"));

  float scale = 0.125f;  // not const: see the note in run_saxpy
  std::vector<uint32_t> packed(words);
  // A fixed LCG so a failure is reproducible from the seed alone.
  uint32_t state = 0x12345678u;
  for (uint32_t i = 0; i < words; ++i) {
    state = state * 1664525u + 1013904223u;
    packed[i] = state;
  }

  const std::vector<float> want = ref_unpack_u4(packed, scale);
  DeviceArray<uint32_t> din(words);
  DeviceArray<float> dout(words * 8);
  din.upload(packed);

  void* args[] = {din.get(), dout.get(), &scale, &words};
  CU_CHECK(cuLaunchKernel(fn, words, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  CU_CHECK(cuCtxSynchronize());

  std::vector<float> got;
  dout.download(got);
  const Diff d = compare(got, want, 0.0);
  if (d.mismatched != 0) {
    std::printf("  unpack_u4    FAIL %zu/%zu differ, max_abs=%g at %zu\n",
                d.mismatched, want.size(), d.max_abs, d.worst);
    return false;
  }
  std::printf("  unpack_u4    ok   %u words -> %zu f32 (scale %g)\n", words,
              got.size(), scale);
  time_it(fn, args, repeat, words, "unpack_u4_f32");
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--repeat" && i + 1 < argc) {
      args.repeat = std::atoi(argv[++i]);
    } else if (flag == "--arch" && i + 1 < argc) {
      args.arch = argv[++i];
    } else {
      args.names.push_back(flag);
    }
  }
  if (args.names.empty()) {
    args.names = {"vec_add_f32", "saxpy_f32", "reduce_sum_f32", "unpack_u4_f32"};
  }

  CU_CHECK(cuInit(0));

  int device_count = 0;
  CU_CHECK(cuDeviceGetCount(&device_count));
  if (device_count == 0) {
    std::fprintf(stderr, "no CUDA device found\n");
    return 1;
  }

  CUdevice device;
  CU_CHECK(cuDeviceGet(&device, 0));

  int driver_major = 0;
  cuDriverGetVersion(&driver_major);
  char name[256] = {};
  cuDeviceGetName(name, sizeof(name), device);

  // Compile for the device we actually have, not for the requested arch: a
  // mismatch here is a confusing "no kernel image" much later on.
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  const std::string device_arch = "sm_" + std::to_string(major) + std::to_string(minor);

  std::printf("device: %s (compute %d.%d)\n", name, major, minor);
  std::printf("driver: %d\n", driver_major);
  std::printf("target: %s (requested %s)\n", device_arch.c_str(), args.arch.c_str());
  std::printf("\n");

  CUcontext context;
  CU_CHECK(cuCtxCreate(&context, 0, device));

  bool all_ok = true;
  for (const std::string& name_arg : args.names) {
    const std::string path = std::string(kCorpus) + "/" + name_arg + ".ptx";
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
      std::printf("  %-12s SKIP no such golden: %s\n", name_arg.c_str(), path.c_str());
      all_ok = false;
      continue;
    }
    std::fseek(f, 0, SEEK_END);
    const long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> text(len + 1);
    const size_t read = std::fread(text.data(), 1, len, f);
    text[read] = '\0';
    std::fclose(f);

    // PTX_JIT_TARGET_ARCH: honour the device rather than whatever was asked for.
    CUjit_option options[] = {CU_JIT_TARGET_ARCH};
    const std::string arch = args.arch == device_arch ? device_arch : device_arch;
    const char* arch_cstr = arch.c_str();
    void* option_values[] = {&arch_cstr};

    CUmodule module;
    CUresult status = cuModuleLoadDataEx(&module, text.data(), 2, options, option_values);
    if (status != CUDA_SUCCESS) {
      const char* err = nullptr;
      cuGetErrorName(status, &err);
      std::printf("  %-12s JIT FAIL: %s (%s)\n", name_arg.c_str(),
                  err ? err : "?", path.c_str());
      all_ok = false;
      continue;
    }

    const uint32_t n = 1u << 20;
    bool ok = false;
    if (name_arg == "vec_add_f32") ok = run_vec_add(module, args.repeat, n);
    else if (name_arg == "saxpy_f32") ok = run_saxpy(module, args.repeat, n);
    else if (name_arg == "reduce_sum_f32") ok = run_reduce(module, args.repeat, 1u << 16);
    else if (name_arg == "unpack_u4_f32") ok = run_unpack(module, args.repeat, 1u << 18);
    else {
      std::printf("  %-12s SKIP no CPU reference implemented\n", name_arg.c_str());
    }

    all_ok = all_ok && ok;
    cuModuleUnload(module);
  }

  std::printf("\n");
  std::printf("%s\n", all_ok ? "all kernels matched their references"
                             : "at least one kernel failed");
  cuCtxDestroy(context);
  return all_ok ? 0 : 1;
}
