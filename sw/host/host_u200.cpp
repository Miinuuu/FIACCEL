// U200 self-test and bounded-memory 1920x1080 RGB24 video streaming.
#include "accelerator.h"
#include "golden_reference.h"
#include "raw_video_output.h"
#include "video_pipeline.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace fiaccel;
int self_test(const std::string& xclbin,const std::string& output) {
    const auto records=validation_config();
    Accelerator device(xclbin,std::cout);
    if(std::getenv("FIACCEL_CONFIG_SMOKE")) {
        device.configuration_smoke(records.front());
        std::cout<<"PASS configuration/control/m00 AXI smoke only; frame processing not run\n";
        return 0;
    }
    device.configure(records);
    device.warmup(Frame(Pixels,0x00dfefffu));
    auto first=device.interpolate(Frame(Pixels,0x00808080u),"Interpolated frame",true);
    if(first.stats.writes!=FeatureVectors || first.stats.skips)throw std::runtime_error("Interpolated-frame selective-transfer counts");
    const auto mask=nonzero_reference(Width,Height);
    unsigned nonzero=0;
    for(unsigned i=0;i<Pixels;++i) {
        if(first.pixels[i]!=expected_rgb(mask[i]))throw std::runtime_error("Nonzero RGB mismatch at pixel "+std::to_string(i));
        nonzero+=mask[i]!=0;
    }
    if(!nonzero)throw std::runtime_error("Reference did not exercise nonzero arithmetic");
    write_ppm(output,first.pixels);
    auto second=device.interpolate(Frame(Pixels,0x00808080u),"Retained-gray frame",true);
    if(second.stats.writes || second.stats.skips!=FeatureVectors)throw std::runtime_error("Retained-gray selective-transfer counts");
    for(auto pixel:second.pixels)if(pixel!=0x00808080u)throw std::runtime_error("Retained-gray RGB mismatch");
    std::cout<<"PASS U200 FIACCEL RGB pixels="<<Pixels<<" output_bytes="<<Pixels*4
             <<" frames_checked=2 nonzero_mask_pixels="<<nonzero<<'\n';
    return 0;
}
int video_stream(const std::string& xclbin,const std::string& config,bool strict,bool serial) {
    RawVideoOutput video_output;
    // Validate every resident coefficient before touching the device.
    const auto records=load_config(config);
    Accelerator device(xclbin,std::cerr);
    device.configure(records);
    Rgb24Reader input(std::cin);Rgb24Writer output(video_output.stream());
    uint32_t max_cycles=0;uint64_t over_budget=0;
    const auto begin=std::chrono::steady_clock::now();
    auto observe=[&](const CommandStats& stats) {
        max_cycles=std::max(max_cycles,stats.cycles);
        over_budget+=stats.cycles>CycleLimit;
    };
    const auto counts=serial ? interpolate_stream(input,output,
        [&](const Frame& first){device.warmup(first);},
        [&](const Frame& current){
            auto result=device.interpolate(current,"",strict);
            observe(result.stats);
            return std::move(result.pixels);
        }) : interpolate_stream_pipelined(input,output,device,observe,strict);
    video_output.stream().flush();
    if(!video_output.stream())throw std::runtime_error("Cannot flush RGB24 output");
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::cerr<<"HOST_PROFILE pipeline="<<(serial ? "serial" : "overlap")<<" stream_seconds="<<seconds
             <<" interpolated_fps="<<counts.interpolated_frames/seconds<<'\n';
    std::cerr<<"VIDEO_SUMMARY input_frames="<<counts.input_frames<<" output_frames="<<counts.output_frames
             <<" interpolated_frames="<<counts.interpolated_frames<<" width="<<Width<<" height="<<Height
             <<" max_kernel_cycles="<<max_cycles<<" over_budget_frames="<<over_budget<<'\n';
    return 0;
}
}
int main(int argc,char** argv) try {
    std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
    if(argc==5 && std::string(argv[1])=="--reference-only") {
        const auto mask=fiaccel::nonzero_reference(std::stoul(argv[2]),std::stoul(argv[3]));
        std::ofstream stream(argv[4],std::ios::binary);
        for(auto value:mask) {
            const uint32_t words[]={uint32_t(value),fiaccel::expected_rgb(value)};
            stream.write(reinterpret_cast<const char*>(words),sizeof(words));
        }
        if(!stream)throw std::runtime_error("Cannot write reference values");
        return 0;
    }
    if(argc>=3 && std::string(argv[2])=="--video-stream") {
        std::string weights;bool strict=false,serial=false;
        for(int i=3;i<argc;++i) {
            const std::string option=argv[i];
            if(option=="--weights" && i+1<argc)weights=argv[++i];
            else if(option=="--enforce-cycle-budget")strict=true;
            else if(option=="--serial")serial=true;
            else throw std::runtime_error("Unknown video-stream option: "+option);
        }
        if(weights.empty())throw std::runtime_error("Video mode requires --weights config.hex");
        return video_stream(argv[1],weights,strict,serial);
    }
    if(argc<2 || argc>3)throw std::runtime_error("Usage: host XCLBIN [output.ppm] | host XCLBIN --video-stream --weights config.hex [--enforce-cycle-budget] [--serial]");
    return self_test(argv[1],argc==3 ? argv[2] : "synthesized.ppm");
} catch(const std::exception& error) {
    std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;
}
