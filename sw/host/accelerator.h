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

// One resident coefficient set, a persistent FMR cache and three recycled
// device image buffers serve the entire stream. No whole-video allocation.
class Accelerator {
    xrt::device device_;
    xrt::uuid uuid_;
    xrt::ip kernel_;
    std::array<xrt::bo,3> images_;
    std::array<xrt::bo,2> outputs_;
    xrt::bo cache_,descriptor_;
    std::array<uint32_t*,3> image_data_;
    std::array<uint32_t*,2> output_data_;
    uint64_t* descriptor_data_;
    std::ostream& log_;
    std::chrono::seconds timeout_;
    unsigned previous_=0,frame_number_=0;
    bool primed_=false;
    bool staged_=false,in_flight_=false,output_ready_=false;
    unsigned staged_slot_=0,running_output_=0,completed_output_=0;
    uint32_t writes_before_=0,skips_before_=0;
    std::chrono::steady_clock::time_point command_begin_;
    std::string command_label_;
    bool command_budget_=false;
    void begin_command(uint32_t mode,uint64_t pointer,const std::string& label,bool enforce_budget);
    CommandStats wait_command();
    CommandStats command(uint32_t mode,uint64_t pointer,const std::string& label,bool enforce_budget);
    void upload(unsigned slot,const Frame& frame);
    void describe(unsigned current,unsigned previous,unsigned output);
    CommandStats process(unsigned current,unsigned previous,const std::string& label,bool enforce_budget);
public:
    explicit Accelerator(const std::string& xclbin,std::ostream& log);
    ~Accelerator();
    Accelerator(const Accelerator&)=delete;
    Accelerator& operator=(const Accelerator&)=delete;
    void configure(const std::vector<uint64_t>& records);
    void configuration_smoke(uint64_t record);
    CommandStats warmup(const Frame& first);
    InterpolatedFrame interpolate(const Frame& current,const std::string& label="",bool enforce_budget=false);
    // Stage into the unused third input while the FPGA reads the other two.
    // Only one command runs at once: FMR representatives remain ordered.
    void stage_frame(const Frame& current);
    void start_interpolation(const std::string& label="",bool enforce_budget=false);
    CommandStats finish_interpolation();
    // The next command may run in the other output BO during this readback.
    void read_output(Frame& pixels);
};
}
