// Per-kernel dispatch cost on gfx1152 under Windows HIP: back-to-back launches, HIP graph replay,
// a ~20 us memory-bound kernel in sequence, and event pairs between kernels. Prints one JSON line per case.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>

#define CHECK(x) do { hipError_t check_err_ = (x); if (check_err_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(check_err_)); return 1; } } while (0)

__global__ void tiny(float * p) { p[threadIdx.x] += 1.0f; }

// Reads n floats with a grid-stride loop; sized so one launch takes on the order of 20 us.
__global__ void stream_read(const float4 * __restrict__ x, float * out, size_t n4) {
    float4 acc = make_float4(0, 0, 0, 0);
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x) {
        const float4 v = x[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x + acc.y + acc.z + acc.w == 1234.5f) { out[0] = acc.x; }
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    hipStream_t s;
    CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    float * p;
    CHECK(hipMalloc(&p, 4096));
    CHECK(hipMemset(p, 0, 4096));
    const size_t bytes = 1600000;  // ~1.6 MB: ~20 us at 80 GB/s
    float4 * x;
    CHECK(hipMalloc(&x, bytes * 64));
    CHECK(hipMemset(x, 0, bytes * 64));
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a));
    CHECK(hipEventCreate(&b));

    auto timed = [&](const char * name, int n, auto && body) -> int {
        for (int w = 0; w < 3; ++w) { body(); }
        CHECK(hipStreamSynchronize(s));
        const double t0 = now_us();
        CHECK(hipEventRecord(a, s));
        for (int r = 0; r < n; ++r) { body(); }
        CHECK(hipEventRecord(b, s));
        CHECK(hipStreamSynchronize(s));
        const double t1 = now_us();
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, a, b));
        printf("{\"case\": \"%s\", \"iterations\": %d, \"wall_us_per_iter\": %.3f, \"gpu_us_per_iter\": %.3f}\n",
               name, n, (t1 - t0) / n, 1000.0 * ms / n);
        fflush(stdout);
        return 0;
    };

    // 1. back-to-back tiny kernels
    if (timed("tiny_kernel", 20000, [&] { tiny<<<1, 64, 0, s>>>(p); })) return 1;

    // 2. the same tiny kernels replayed from one captured graph of 1000 launches (time per kernel = /1000)
    hipGraph_t g;
    hipGraphExec_t ge;
    CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeRelaxed));
    for (int i = 0; i < 1000; ++i) { tiny<<<1, 64, 0, s>>>(p); }
    CHECK(hipStreamEndCapture(s, &g));
    CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    if (timed("tiny_kernel_graph_x1000", 20, [&] { (void) hipGraphLaunch(ge, s); })) return 1;

    // 3. ~20 us streaming kernels: one launch vs. distinct buffers back-to-back (so caches do not help)
    const size_t n4 = bytes / sizeof(float4);
    int slice = 0;
    if (timed("stream_1.6MB", 2000, [&] {
        stream_read<<<64, 256, 0, s>>>(x + (slice++ % 64) * n4, p, n4); })) return 1;

    // 4. tiny kernels with an event pair around each (what an event-based op profiler adds)
    std::vector<hipEvent_t> ev(2 * 2000);
    for (auto & e : ev) { CHECK(hipEventCreate(&e)); }
    int k = 0;
    if (timed("tiny_kernel_with_event_pair", 2000, [&] {
        (void) hipEventRecord(ev[(2 * k) % ev.size()], s);
        tiny<<<1, 64, 0, s>>>(p);
        (void) hipEventRecord(ev[(2 * k + 1) % ev.size()], s);
        ++k; })) return 1;
    return 0;
}
