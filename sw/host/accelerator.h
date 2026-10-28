#pragma once
#include "frame_io.h"
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/experimental/xrt_ip.h>
#include <array>
#include <chrono>
#include <iosfwd>

namespace fiaccel {
struct CommandStats {
    uint32_t cycles=0,writes=0,skips=0;
    double seconds=0;
};
struct InterpolatedFrame { Frame pixels;CommandStats stats; };

// One resident coefficient set, a persistent FMR cache and two recycled
// device image buffers serve the entire stream. No whole-video allocation.
class Accelerator {
    xrt::device device_;
    xrt::uuid uuid_;
    xrt::ip kernel_;
    std::array<xrt::bo,2> images_;
    xrt::bo output_,cache_,descriptor_;
    std::ostream& log_;
    std::chrono::seconds timeout_;
    unsigned previous_=0,frame_number_=0;
    bool primed_=false;
    CommandStats command(uint32_t mode,uint64_t pointer,const std::string& label,bool enforce_budget);
    void upload(unsigned slot,const Frame& frame);
    CommandStats process(unsigned current,unsigned previous,const std::string& label,bool enforce_budget);
public:
    explicit Accelerator(const std::string& xclbin,std::ostream& log);
    void configure(const std::vector<uint64_t>& records);
    void configuration_smoke(uint64_t record);
    CommandStats warmup(const Frame& first);
    InterpolatedFrame interpolate(const Frame& current,const std::string& label="",bool enforce_budget=false);
};
}
