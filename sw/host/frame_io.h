#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

namespace fiaccel {
inline constexpr unsigned Width=1920,Height=1080,Pixels=Width*Height;
inline constexpr unsigned FeatureVectors=Pixels/64,CycleLimit=6220800;
inline constexpr std::array<unsigned,12> WeightCounts={216,1152,4608,36864,36864,36864,36864,16384,18432,4608,1152,216};
using Frame=std::vector<uint32_t>;
std::vector<uint64_t> validation_config();
std::vector<uint64_t> load_config(const std::string& path);
void write_ppm(const std::string& path,const Frame& frame,unsigned width=Width,unsigned height=Height);

class Rgb24Reader {
    std::istream& stream_;
    std::vector<char> bytes_;
public:
    explicit Rgb24Reader(std::istream& stream,std::size_t pixels=Pixels);
    bool read(Frame& frame);
};
class Rgb24Writer {
    std::ostream& stream_;
    std::vector<char> bytes_;
public:
    explicit Rgb24Writer(std::ostream& stream,std::size_t pixels=Pixels);
    void write(const Frame& frame);
};
struct StreamCounts { uint64_t input_frames=0,output_frames=0,interpolated_frames=0; };
// Emit F0, midpoint(F0,F1), F1, ... and hold the last frame once.
// N inputs produce 2N outputs at twice the input FPS, preserving duration.
StreamCounts interpolate_stream(Rgb24Reader& input,Rgb24Writer& output,
    const std::function<void(const Frame&)>& warmup,
    const std::function<Frame(const Frame&)>& interpolate);
}
