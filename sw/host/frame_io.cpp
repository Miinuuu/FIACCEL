#include "frame_io.h"
#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace fiaccel {
std::vector<uint64_t> validation_config() {
    std::vector<uint64_t> records;
    for(unsigned layer=0;layer<WeightCounts.size();++layer) {
        for(unsigned address=0;address<WeightCounts[layer];++address)
            records.push_back((uint64_t(layer)<<43)|(uint64_t(address)<<11)|1);
        records.push_back((uint64_t(1)<<47)|(uint64_t(layer)<<43)|(layer<3 || layer==11 ? 0 : 8));
    }
    return records;
}
std::vector<uint64_t> load_config(const std::string& path) {
    std::ifstream input(path);
    if(!input)throw std::runtime_error("Cannot open coefficient config: "+path);
    std::array<std::vector<bool>,12> seen;
    for(unsigned i=0;i<seen.size();++i)seen[i].resize(WeightCounts[i]);
    std::array<bool,12> shifts{};
    std::vector<uint64_t> records;
    std::string line;
    while(std::getline(input,line)) {
        line=line.substr(0,line.find('#'));
        std::istringstream words(line);
        std::string word,extra;
        if(!(words>>word))continue;
        if(words>>extra)throw std::runtime_error("Expected one configuration word per line");
        std::size_t used=0;
        const uint64_t record=std::stoull(word,&used,16);
        if(used!=word.size() || record>>48 || (record&0x700))throw std::runtime_error("Invalid 48-bit configuration word");
        const unsigned layer=(record>>43)&15,address=(record>>11)&0xffffffffu;
        if(layer>=12)throw std::runtime_error("Invalid coefficient layer");
        if(record&(uint64_t(1)<<47)) {
            if(address!=0 || (record&255)>63 || shifts[layer])throw std::runtime_error("Invalid or duplicate quantization shift");
            shifts[layer]=true;
        } else {
            if(address>=seen[layer].size() || seen[layer][address])throw std::runtime_error("Invalid or duplicate coefficient address");
            seen[layer][address]=true;
        }
        records.push_back(record);
    }
    if(input.bad())throw std::runtime_error("Cannot read coefficient config");
    for(unsigned i=0;i<seen.size();++i)
        if(!shifts[i] || std::find(seen[i].begin(),seen[i].end(),false)!=seen[i].end())
            throw std::runtime_error("Config must contain every INT8 coefficient and all 12 shifts");
    return records;
}
Rgb24Reader::Rgb24Reader(std::istream& stream,std::size_t pixels):stream_(stream),bytes_(pixels*3) {
    if(!pixels)throw std::runtime_error("Frame size must be positive");
}
bool Rgb24Reader::read(Frame& frame) {
    stream_.read(bytes_.data(),std::streamsize(bytes_.size()));
    if(stream_.gcount()==0 && stream_.eof())return false;
    if(std::size_t(stream_.gcount())!=bytes_.size())throw std::runtime_error("Truncated RGB24 input frame");
    frame.resize(bytes_.size()/3);
    for(std::size_t i=0;i<frame.size();++i)
        frame[i]=uint8_t(bytes_[3*i])|(uint32_t(uint8_t(bytes_[3*i+1]))<<8)|(uint32_t(uint8_t(bytes_[3*i+2]))<<16);
    return true;
}
Rgb24Writer::Rgb24Writer(std::ostream& stream,std::size_t pixels):stream_(stream),bytes_(pixels*3) {}
void Rgb24Writer::write(const Frame& frame) {
    if(frame.size()*3!=bytes_.size())throw std::runtime_error("Output frame size mismatch");
    for(std::size_t i=0;i<frame.size();++i)for(unsigned c=0;c<3;++c)bytes_[3*i+c]=char(frame[i]>>(8*c));
    stream_.write(bytes_.data(),std::streamsize(bytes_.size()));
    if(!stream_)throw std::runtime_error("Cannot write RGB24 output");
}
void write_ppm(const std::string& path,const Frame& frame,unsigned w,unsigned h) {
    std::ofstream output(path,std::ios::binary);
    if(!output)throw std::runtime_error("Cannot create output image: "+path);
    output<<"P6\n"<<w<<' '<<h<<"\n255\n";
    Rgb24Writer writer(output,std::size_t(w)*h);writer.write(frame);
}
StreamCounts interpolate_stream(Rgb24Reader& input,Rgb24Writer& output,
    const std::function<void(const Frame&)>& warmup,
    const std::function<Frame(const Frame&)>& interpolate) {
    Frame frame;
    if(!input.read(frame))throw std::runtime_error("Input video contains no complete frames");
    warmup(frame);output.write(frame);
    StreamCounts counts{1,1,0};
    while(input.read(frame)) {
        output.write(interpolate(frame));output.write(frame);
        ++counts.input_frames;++counts.interpolated_frames;counts.output_frames+=2;
    }
    output.write(frame);++counts.output_frames;
    return counts;
}
}
