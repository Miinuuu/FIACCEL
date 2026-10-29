#include "video_pipeline.h"
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace fiaccel;
void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
// Independent device substitute: every channel is the average of the two
// input pixels. Keep running and completed results separate, like two BOs.
struct Device {
    unsigned previous=0,staged=0,running_pixel=0,completed_pixel=0;
    bool ready=false,running=false,staged_valid=false;
    unsigned staged_during_run=0,read_during_run=0,finished=0;
    bool expect_strict=false;
    void warmup(const Frame& frame) {previous=frame[0];}
    void stage_frame(const Frame& frame) {
        require(!staged_valid,"Staged input overwritten");
        staged=frame[0];staged_valid=true;staged_during_run+=running;
    }
    void start_interpolation(const std::string&,bool strict) {
        require(staged_valid && !running,"Missing input or overlapping commands");
        require(strict==expect_strict,"Strict flag lost");
        running_pixel=(previous+staged)/2;previous=staged;
        running=true;staged_valid=false;
    }
    unsigned finish_interpolation() {
        require(running && !ready,"Completed output overwritten");
        completed_pixel=running_pixel;running=false;ready=true;return ++finished;
    }
    void read_output(Frame& frame) {
        require(ready,"Output read before completion");
        frame.assign(1,completed_pixel);ready=false;read_during_run+=running;
    }
};
int main() try {
    for(unsigned n:{1u,2u,3u,9u})for(bool strict:{false,true}) {
        std::ostringstream bytes;Rgb24Writer source(bytes,1);
        for(unsigned i=0;i<n;++i)source.write(Frame{i*20});
        std::istringstream input(bytes.str());Rgb24Reader reader(input,1);
        std::ostringstream actual;Rgb24Writer writer(actual,1);
        Device device;device.expect_strict=strict;unsigned seen=0;
        auto counts=interpolate_stream_pipelined(reader,writer,device,[&](unsigned completed) {
            require(completed==++seen,"Completion callback order");
        },strict);
        std::ostringstream expected;Rgb24Writer expected_writer(expected,1);
        expected_writer.write(Frame{0});
        for(unsigned i=1;i<n;++i) {
            expected_writer.write(Frame{i*20-10});expected_writer.write(Frame{i*20});
        }
        expected_writer.write(Frame{(n-1)*20});
        require(actual.str()==expected.str(),"Midpoints, originals and final hold must be exact");
        require(counts.input_frames==n && counts.output_frames==2*n && counts.interpolated_frames==n-1,"Frame accounting");
        require(seen==n-1 && !device.running && !device.ready,"Pending output at EOF");
        require(device.staged_during_run==(n>2 ? n-2 : 0),"Input prefetch not overlapped");
        require(device.read_during_run==(n>2 ? n-2 : 0),"Output readback not overlapped");
    }
    for(const std::string& invalid:{std::string(),std::string("ab"),std::string(7,'x')}) {
        std::istringstream input(invalid);Rgb24Reader reader(input,1);
        std::ostringstream actual;Rgb24Writer writer(actual,1);Device device;bool failed=false;
        try {interpolate_stream_pipelined(reader,writer,device,[](unsigned){});}
        catch(const std::runtime_error&) {failed=true;}
        require(failed,"Empty or truncated input accepted");
    }
    std::cout<<"PASS ordered overlap, pending-output ownership, EOF, strict flag and frame accounting\n";
}catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
