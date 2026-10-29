#include "frame_io.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace fiaccel;
void check(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
template<class Function> void rejects(Function function) {
    try {function();}catch(const std::exception&) {return;}
    throw std::runtime_error("Expected malformed input to fail");
}
int main(int argc,char** argv) try {
    check(argc==2,"Expected a temporary config path");
    std::ostringstream encoded;
    Rgb24Writer writer(encoded,2);writer.write({0x00332211,0x00ccbbaa});
    check(encoded.str()==std::string("\x11\x22\x33\xaa\xbb\xcc",6),"RGB packing");
    std::istringstream bytes(encoded.str());Rgb24Reader reader(bytes,2);Frame frame;
    check(reader.read(frame) && frame==Frame({0x00332211,0x00ccbbaa}),"RGB unpacking");
    check(!reader.read(frame) && frame.size()==2,"Clean EOF must retain the last frame");
    std::istringstream partial("abcde");Rgb24Reader truncated(partial,2);
    rejects([&]{truncated.read(frame);});
    rejects([&]{writer.write({1});});
    for(unsigned n:{1u,3u}) {
        std::ostringstream source;
        Rgb24Writer source_writer(source,1);
        for(unsigned i=0;i<n;++i)source_writer.write({i*20});
        std::istringstream stream(source.str());Rgb24Reader input(stream,1);
        std::ostringstream sink;Rgb24Writer output(sink,1);unsigned previous=0,warms=0,calls=0;
        auto counts=interpolate_stream(input,output,[&](const Frame& f){previous=f[0];++warms;},
            [&](const Frame& f){const auto midpoint=(previous+f[0])/2;previous=f[0];++calls;return Frame{midpoint};});
        check(counts.input_frames==n && counts.output_frames==2*n && counts.interpolated_frames==n-1,"Frame counts");
        check(warms==1 && calls==n-1,"Warmup/interpolation calls");
        const std::string expected=n==1 ? std::string(6,'\0') : std::string("\0\0\0\x0a\0\0\x14\0\0\x1e\0\0\x28\0\0\x28\0\0",18);
        check(sink.str()==expected,"Frame order and final hold");
    }
    std::istringstream empty;Rgb24Reader no_frames(empty,1);
    rejects([&]{interpolate_stream(no_frames,writer,[](const Frame&){},[](const Frame& f){return f;});});
    const std::string path=argv[1];
    auto write_config=[&](const std::vector<uint64_t>& records){std::ofstream out(path);out<<std::hex;for(auto r:records)out<<r<<'\n';};
    auto records=validation_config();write_config(records);
    check(load_config(path)==records,"Complete resident configuration");
    {
        std::ofstream out(path);out<<"# comment\n\n"<<std::hex;
        for(auto r:records)out<<" \t+0x"<<r<<" \t# coefficient\r\n";
    }
    check(load_config(path)==records,"Comments, whitespace and hexadecimal prefix");
    for(const auto* malformed:{"+","0x","-1","1000000000000","not_hex","1 2","fffffffffffffffff"}) {
        {std::ofstream out(path);out<<malformed<<'\n';}
        rejects([&]{load_config(path);});
    }
    records.push_back(records.front());write_config(records);rejects([&]{load_config(path);});
    records=validation_config();records.pop_back();write_config(records);rejects([&]{load_config(path);});
    records=validation_config();records.front()=uint64_t(12)<<43;write_config(records);rejects([&]{load_config(path);});
    records=validation_config();records.back()=(uint64_t(1)<<47)|(uint64_t(11)<<43)|64;write_config(records);rejects([&]{load_config(path);});
    std::cout<<"PASS RGB24 framing, frame order, EOF and full INT8 configuration validation\n";
}catch(const std::exception& error){std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
