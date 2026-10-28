#include "accelerator.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace fiaccel {
Accelerator::Accelerator(const std::string& xclbin,std::ostream& log):
    device_(0),uuid_(device_.load_xclbin(xclbin)),kernel_(device_,uuid_,"FIACCEL:{FIACCEL_1}"),
    images_{{xrt::bo(device_,Pixels*4,0),xrt::bo(device_,Pixels*4,0)}},
    output_(device_,Pixels*4,0),cache_(device_,FeatureVectors*32,1),descriptor_(device_,32,0),
    log_(log),timeout_(std::getenv("FIACCEL_TIMEOUT_SECONDS") ? std::stoul(std::getenv("FIACCEL_TIMEOUT_SECONDS")) : 3600) {
    kernel_.write_register(0x18,uint32_t(cache_.address()));kernel_.write_register(0x1c,uint32_t(cache_.address()>>32));
    kernel_.write_register(0x3c,uint32_t(output_.address()));kernel_.write_register(0x40,uint32_t(output_.address()>>32));
}
CommandStats Accelerator::command(uint32_t mode,uint64_t pointer,const std::string& label,bool enforce_budget) {
    const uint32_t writes_before=kernel_.read_register(0x34),skips_before=kernel_.read_register(0x38);
    kernel_.write_register(0x10,mode);
    kernel_.write_register(0x24,uint32_t(pointer));kernel_.write_register(0x28,uint32_t(pointer>>32));
    const auto begin=std::chrono::steady_clock::now();
    kernel_.write_register(0,1);
    while(true) {
        const auto status=kernel_.read_register(0);
        if(kernel_.read_register(0x30))throw std::runtime_error(label+": RTL/AXI error");
        if(status&2)break;
        if(std::chrono::steady_clock::now()-begin>timeout_)throw std::runtime_error(label+": timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CommandStats result;
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    result.cycles=kernel_.read_register(0x44);
    result.writes=kernel_.read_register(0x34)-writes_before;
    result.skips=kernel_.read_register(0x38)-skips_before;
    log_<<label<<" completed in "<<result.seconds<<" s nominal_300MHz_wall_cycle_upper_bound="
        <<uint64_t(result.seconds*300000000.+1.)<<" kernel_cycles="<<result.cycles<<'\n'<<std::flush;
    if(!result.cycles)throw std::runtime_error(label+": missing hardware cycle readback");
    if(enforce_budget && result.cycles>CycleLimit)throw std::runtime_error(label+": hardware exceeds 6.2208M cycles");
    return result;
}
void Accelerator::configure(const std::vector<uint64_t>& records) {
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
    std::copy(frame.begin(),frame.end(),images_[slot].map<uint32_t*>());
    images_[slot].sync(XCL_BO_SYNC_BO_TO_DEVICE);
}
CommandStats Accelerator::process(unsigned current,unsigned previous,const std::string& label,bool enforce_budget) {
    auto descriptor=descriptor_.map<uint64_t*>();
    descriptor[0]=images_[current].address();descriptor[1]=images_[previous].address();
    descriptor[2]=output_.address();descriptor[3]=cache_.address();
    descriptor_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return command(0,descriptor_.address(),label,enforce_budget);
}
CommandStats Accelerator::warmup(const Frame& first) {
    if(primed_)throw std::runtime_error("Stream already primed");
    upload(previous_,first);
    auto stats=process(previous_,previous_,"Warm-up frame",false);
    if(stats.writes!=FeatureVectors || stats.skips)throw std::runtime_error("Warm-up selective-transfer counts");
    primed_=true;return stats;
}
InterpolatedFrame Accelerator::interpolate(const Frame& current,const std::string& label,bool enforce_budget) {
    if(!primed_)throw std::runtime_error("Warm-up frame is required before interpolation");
    const unsigned next=previous_^1;
    upload(next,current);
    auto stats=process(next,previous_,label.empty() ? "Video frame "+std::to_string(++frame_number_) : label,enforce_budget);
    if(uint64_t(stats.writes)+stats.skips!=FeatureVectors)throw std::runtime_error("Per-frame selective-transfer accounting");
    output_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const auto pixels=output_.map<uint32_t*>();
    previous_=next;
    return {Frame(pixels,pixels+Pixels),stats};
}
}
