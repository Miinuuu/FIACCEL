#include "accelerator.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <thread>

namespace fiaccel {
Accelerator::Accelerator(const std::string& xclbin,std::ostream& log):
    device_(0),uuid_(device_.load_xclbin(xclbin)),kernel_(device_,uuid_,"FIACCEL:{FIACCEL_1}"),
    images_{{xrt::bo(device_,Pixels*4,0),xrt::bo(device_,Pixels*4,0),xrt::bo(device_,Pixels*4,0)}},
    outputs_{{xrt::bo(device_,Pixels*4,0),xrt::bo(device_,Pixels*4,0)}},cache_(device_,FeatureVectors*32,1),descriptor_(device_,32,0),
    log_(log),timeout_(std::getenv("FIACCEL_TIMEOUT_SECONDS") ? std::stoul(std::getenv("FIACCEL_TIMEOUT_SECONDS")) : 3600) {
    for(unsigned i=0;i<images_.size();++i)image_data_[i]=images_[i].map<uint32_t*>();
    for(unsigned i=0;i<outputs_.size();++i)output_data_[i]=outputs_[i].map<uint32_t*>();
    descriptor_data_=descriptor_.map<uint64_t*>();
    kernel_.write_register(0x18,uint32_t(cache_.address()));kernel_.write_register(0x1c,uint32_t(cache_.address()>>32));
    kernel_.write_register(0x3c,uint32_t(outputs_[0].address()));kernel_.write_register(0x40,uint32_t(outputs_[0].address()>>32));
}
Accelerator::~Accelerator() {
    // Drain before releasing BOs if a stream I/O exception unwinds a command.
    if(in_flight_)try {wait_command();}catch(const std::exception&) {}
}
void Accelerator::begin_command(uint32_t mode,uint64_t pointer,const std::string& label,bool enforce_budget) {
    writes_before_=kernel_.read_register(0x34);skips_before_=kernel_.read_register(0x38);
    command_label_=label;command_budget_=enforce_budget;
    kernel_.write_register(0x10,mode);
    kernel_.write_register(0x24,uint32_t(pointer));kernel_.write_register(0x28,uint32_t(pointer>>32));
    command_begin_=std::chrono::steady_clock::now();
    kernel_.write_register(0,1);
}
CommandStats Accelerator::wait_command() {
    const auto& label=command_label_;
    unsigned polls=0;
    while(true) {
        const auto status=kernel_.read_register(0);
        if(status&2)break;
        // Check faults at roughly the original 1 ms cadence while polling
        // completion more often. A fault must not wait for the full timeout.
        if(polls++%10==0 && kernel_.read_register(0x30)) {
            in_flight_=false;
            throw std::runtime_error(label+": RTL/AXI error");
        }
        if(std::chrono::steady_clock::now()-command_begin_>timeout_)throw std::runtime_error(label+": timeout");
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    in_flight_=false;
    if(kernel_.read_register(0x30))throw std::runtime_error(label+": RTL/AXI error");
    CommandStats result;
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-command_begin_).count();
    result.cycles=kernel_.read_register(0x44);
    result.writes=kernel_.read_register(0x34)-writes_before_;
    result.skips=kernel_.read_register(0x38)-skips_before_;
    std::ostringstream message;
    message<<label<<" completed in "<<result.seconds<<" s nominal_300MHz_wall_cycle_upper_bound="
        <<uint64_t(result.seconds*300000000.+1.)<<" kernel_cycles="<<result.cycles<<'\n';
    log_<<message.str()<<std::flush;
    if(!result.cycles)throw std::runtime_error(label+": missing hardware cycle readback");
    if(command_budget_ && result.cycles>CycleLimit)throw std::runtime_error(label+": hardware exceeds 6.2208M cycles");
    return result;
}
CommandStats Accelerator::command(uint32_t mode,uint64_t pointer,const std::string& label,bool enforce_budget) {
    begin_command(mode,pointer,label,enforce_budget);
    return wait_command();
}
void Accelerator::configure(const std::vector<uint64_t>& records) {
    if(in_flight_ || staged_ || output_ready_)throw std::runtime_error("Cannot configure an active stream");
    if(records.empty() || records.size()>=0x80000000u)throw std::runtime_error("Invalid configuration size");
    auto config=xrt::bo(device_,records.size()*8,0);
    std::copy(records.begin(),records.end(),config.map<uint64_t*>());
    config.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    command(0x80000000u|uint32_t(records.size()),config.address(),"Weight configuration",false);
    primed_=false;previous_=0;frame_number_=0;
}
void Accelerator::configuration_smoke(uint64_t record) {
    auto config=xrt::bo(device_,32,0);config.map<uint64_t*>()[0]=record;
    config.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    command(0x80000001u,config.address(),"Single-record configuration smoke",false);
}
void Accelerator::upload(unsigned slot,const Frame& frame) {
    if(frame.size()!=Pixels)throw std::runtime_error("Kernel requires 1920x1080 RGB input");
    std::copy(frame.begin(),frame.end(),image_data_[slot]);
    images_[slot].sync(XCL_BO_SYNC_BO_TO_DEVICE);
}
void Accelerator::describe(unsigned current,unsigned previous,unsigned output) {
    descriptor_data_[0]=images_[current].address();descriptor_data_[1]=images_[previous].address();
    descriptor_data_[2]=outputs_[output].address();descriptor_data_[3]=cache_.address();
    descriptor_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    kernel_.write_register(0x3c,uint32_t(outputs_[output].address()));
    kernel_.write_register(0x40,uint32_t(outputs_[output].address()>>32));
}
CommandStats Accelerator::process(unsigned current,unsigned previous,const std::string& label,bool enforce_budget) {
    describe(current,previous,0);
    return command(0,descriptor_.address(),label,enforce_budget);
}
CommandStats Accelerator::warmup(const Frame& first) {
    if(primed_)throw std::runtime_error("Stream already primed");
    upload(previous_,first);
    auto stats=process(previous_,previous_,"Warm-up frame",false);
    if(stats.writes!=FeatureVectors || stats.skips)throw std::runtime_error("Warm-up selective-transfer counts");
    primed_=true;return stats;
}
void Accelerator::stage_frame(const Frame& current) {
    if(!primed_ || staged_)throw std::runtime_error("Cannot stage an unprimed or already staged frame");
    staged_slot_=(previous_+1)%images_.size();
    upload(staged_slot_,current);
    staged_=true;
}
void Accelerator::start_interpolation(const std::string& label,bool enforce_budget) {
    if(!staged_ || in_flight_)throw std::runtime_error("Interpolation requires one staged frame and an idle kernel");
    // Preserve a completed output until its host readback has consumed it.
    running_output_=output_ready_ ? completed_output_^1 : frame_number_%outputs_.size();
    describe(staged_slot_,previous_,running_output_);
    ++frame_number_;
    begin_command(0,descriptor_.address(),label.empty() ? "Video frame "+std::to_string(frame_number_) : label,enforce_budget);
    previous_=staged_slot_;staged_=false;in_flight_=true;
}
CommandStats Accelerator::finish_interpolation() {
    if(!in_flight_ || output_ready_)throw std::runtime_error("No running frame or previous output not consumed");
    auto stats=wait_command();
    if(uint64_t(stats.writes)+stats.skips!=FeatureVectors)throw std::runtime_error("Per-frame selective-transfer accounting");
    completed_output_=running_output_;output_ready_=true;in_flight_=false;
    return stats;
}
void Accelerator::read_output(Frame& pixels) {
    if(!output_ready_)throw std::runtime_error("No completed output to read");
    outputs_[completed_output_].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    pixels.resize(Pixels);
    std::copy_n(output_data_[completed_output_],Pixels,pixels.begin());
    output_ready_=false;
}
InterpolatedFrame Accelerator::interpolate(const Frame& current,const std::string& label,bool enforce_budget) {
    stage_frame(current);start_interpolation(label,enforce_budget);
    auto stats=finish_interpolation();Frame pixels;read_output(pixels);
    return {std::move(pixels),stats};
}
}
