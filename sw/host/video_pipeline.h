#pragma once
#include "frame_io.h"
#include <stdexcept>

namespace fiaccel {
// One ordered kernel command; upload into the third input while the other
// two are read. Alternate output BOs for readback during the next command.
// Host storage stays at three frames, independently of video length.
template<class Device,class Observe>
StreamCounts interpolate_stream_pipelined(Rgb24Reader& input,Rgb24Writer& output,
    Device& device,Observe observe,bool strict=false) {
    Frame current,next,midpoint;
    if(!input.read(current))throw std::runtime_error("Input video contains no complete frames");
    device.warmup(current);output.write(current);
    StreamCounts counts{1,1,0};
    if(!input.read(current)) {
        output.write(current);++counts.output_frames;return counts;
    }
    ++counts.input_frames;
    device.stage_frame(current);device.start_interpolation("",strict);
    while(true) {
        const bool more=input.read(next);
        if(more) {device.stage_frame(next);++counts.input_frames;}
        observe(device.finish_interpolation());
        if(more)device.start_interpolation("",strict);
        device.read_output(midpoint);
        output.write(midpoint);output.write(current);
        ++counts.interpolated_frames;counts.output_frames+=2;
        if(!more)break;
        current.swap(next);
    }
    output.write(current);++counts.output_frames;
    return counts;
}
}
