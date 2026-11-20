#include "SDK/ADLXHelper/Windows/Cpp/ADLXHelper.h"
#include "SDK/Include/IPerformanceMonitoring.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <thread>

using namespace adlx;

static void number(ADLX_RESULT result, double value) {
    if (ADLX_SUCCEEDED(result)) std::printf(",%.2f", value);
    else std::printf(",");
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    int interval = std::atoi(argv[1]);
    ADLXHelper helper;
    if (ADLX_FAILED(helper.Initialize())) return 3;
    IADLXGPUListPtr list;
    IADLXPerformanceMonitoringServicesPtr service;
    if (ADLX_FAILED(helper.GetSystemServices()->GetGPUs(&list)) ||
        ADLX_FAILED(helper.GetSystemServices()->GetPerformanceMonitoringServices(&service))) return 4;
    IADLXGPUPtr gpu;
    for (auto i = list->Begin(); i != list->End(); ++i) {
        IADLXGPUPtr candidate;
        const char* name = nullptr;
        if (ADLX_SUCCEEDED(list->At(i, &candidate)) && ADLX_SUCCEEDED(candidate->Name(&name))
            && name && std::strstr(name, "860M")) { gpu = candidate; break; }
    }
    if (!gpu) return 5;
    std::puts("unix_ms,gpu_clock_mhz,gpu_usage_percent,gpu_temperature_c,gpu_power_w,board_power_w,memory_clock_mhz");
    do {
        IADLXGPUMetricsPtr sample;
        if (ADLX_FAILED(service->GetCurrentGPUMetrics(gpu, &sample))) return 6;
        auto now = std::chrono::system_clock::now().time_since_epoch();
        std::printf("%lld", static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count()));
        adlx_int integer = 0;
        adlx_double value = 0;
        auto status = sample->GPUClockSpeed(&integer); number(status, integer);
        status = sample->GPUUsage(&value); number(status, value);
        status = sample->GPUTemperature(&value); number(status, value);
        status = sample->GPUPower(&value); number(status, value);
        status = sample->GPUTotalBoardPower(&value); number(status, value);
        status = sample->GPUVRAMClockSpeed(&integer); number(status, integer);
        std::putchar('\n');
        std::fflush(stdout);
        if (interval <= 0) break;
        std::this_thread::sleep_for(std::chrono::seconds(interval));
    } while (!std::filesystem::exists(argv[2]));
}
