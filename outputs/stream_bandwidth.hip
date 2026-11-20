#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(hipError_t status, const char *call) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", call, hipGetErrorString(status));
        std::exit(1);
    }
}
#define HIP(call) check(call, #call)

constexpr unsigned multiplier = 1664525u, offset = 1013904223u;
constexpr int threads = 256, passes = 6, trials = 7;

__global__ void initialize(uint4 *data, unsigned count) {
    for (unsigned j = blockIdx.x * blockDim.x + threadIdx.x; j < count; j += gridDim.x * blockDim.x) {
        unsigned first = multiplier * (4u * j) + offset;
        data[j] = make_uint4(first, first + multiplier, first + 2u * multiplier, first + 3u * multiplier);
    }
}

__global__ void stream_read(const uint4 *__restrict__ data, uint4 *__restrict__ output, unsigned count) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x;
    uint4 sum = make_uint4(0, 0, 0, 0);
    for (unsigned j = t; j < count; j += 4u * stride) {
        uint4 a = data[j], b = data[j + stride], c = data[j + 2u * stride], d = data[j + 3u * stride];
        sum.x += (a.x + b.x) + (c.x + d.x);
        sum.y += (a.y + b.y) + (c.y + d.y);
        sum.z += (a.z + b.z) + (c.z + d.z);
        sum.w += (a.w + b.w) + (c.w + d.w);
    }
    output[t] = sum;
}

static void validate(uint4 *device_output, unsigned count, unsigned workers) {
    std::vector<uint4> sums(workers);
    HIP(hipMemcpy(sums.data(), device_output, workers * sizeof(uint4), hipMemcpyDeviceToHost));
    for (unsigned t = 0; t < workers; ++t) {
        const uint32_t lanes[] = {sums[t].x, sums[t].y, sums[t].z, sums[t].w};
        uint64_t terms = 1 + (count - 1 - t) / workers;
        uint64_t pairs = terms * (terms - 1) / 2;
        for (unsigned k = 0; k < 4; ++k) {
            uint32_t start = multiplier * (4u * t + k) + offset;
            uint32_t step = multiplier * (4u * workers);
            uint32_t expected = static_cast<uint32_t>(terms * start + pairs * step);
            if (expected != lanes[k]) {
                std::fprintf(stderr, "validation failed at worker %u lane %u\n", t, k);
                std::exit(2);
            }
        }
    }
}

static void measure(unsigned mib, unsigned blocks) {
    size_t bytes = size_t(mib) << 20;
    unsigned count = static_cast<unsigned>(bytes / sizeof(uint4));
    if (count % (4u * blocks * threads) != 0) std::exit(5);
    uint4 *data, *output;
    HIP(hipMalloc(&data, bytes));
    HIP(hipMalloc(&output, blocks * threads * sizeof(uint4)));
    initialize<<<blocks, threads>>>(data, count);
    HIP(hipGetLastError());
    for (int i = 0; i < 3; ++i) {
        stream_read<<<blocks, threads>>>(data, output, count);
        HIP(hipGetLastError());
    }
    HIP(hipDeviceSynchronize());
    hipEvent_t start, end;
    HIP(hipEventCreate(&start));
    HIP(hipEventCreate(&end));
    for (int trial = 1; trial <= trials; ++trial) {
        // Poison every output after warmup so a skipped timed launch cannot pass validation.
        HIP(hipMemset(output, 0xa5, blocks * threads * sizeof(uint4)));
        HIP(hipDeviceSynchronize());
        auto wall_start = std::chrono::steady_clock::now();
        HIP(hipEventRecord(start));
        for (int i = 0; i < passes; ++i) {
            stream_read<<<blocks, threads>>>(data, output, count);
            HIP(hipGetLastError());
        }
        HIP(hipEventRecord(end));
        HIP(hipEventSynchronize(end));
        double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
        float event_ms;
        HIP(hipEventElapsedTime(&event_ms, start, end));
        if (!(event_ms > 0 && wall_ms > 0)) std::exit(3);
        validate(output, count, blocks * threads);
        double mb = double(bytes) * passes / 1e6;
        std::printf("{\"input_mib\":%u,\"blocks\":%u,\"threads\":%d,\"passes\":%d,\"trial\":%d,"
                    "\"event_ms\":%.6f,\"wall_ms\":%.6f,\"event_gb_s\":%.4f,\"wall_gb_s\":%.4f,\"validated\":true}\n",
                    mib, blocks, threads, passes, trial, event_ms, wall_ms, mb / event_ms, mb / wall_ms);
        std::fflush(stdout);
    }
    HIP(hipEventDestroy(start));
    HIP(hipEventDestroy(end));
    HIP(hipFree(data));
    HIP(hipFree(output));
}

int main() {
    hipDeviceProp_t props{};
    HIP(hipGetDeviceProperties(&props, 0));
    if (std::strncmp(props.gcnArchName, "gfx1152", 7) != 0) return 4;
    std::printf("{\"arch\":\"%s\",\"device\":\"%s\",\"compute_units\":%d}\n",
                props.gcnArchName, props.name, props.multiProcessorCount);
    measure(512, 128);
    measure(1024, 512);
    measure(1024, 128);
    measure(512, 512);
}
