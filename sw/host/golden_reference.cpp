#include "golden_reference.h"
#include <algorithm>
#include <stdexcept>
namespace fiaccel {
using Plane=std::vector<int>;
// All-one weights make channels identical. This spatial reference retains
// every padding, integer shift and clamp, independently of the RTL scheduler.
Plane convolution(const Plane& input,unsigned w,unsigned h,unsigned channels,
                  unsigned stride,unsigned dilation,unsigned shift,int limit=255) {
    const unsigned ow=(w+stride-1)/stride,oh=(h+stride-1)/stride;
    Plane output(ow*oh);
    for(unsigned y=0;y<oh;++y)for(unsigned x=0;x<ow;++x) {
        int sum=0;
        for(int ky=-1;ky<=1;++ky)for(int kx=-1;kx<=1;++kx) {
            const int iy=int(y*stride)+ky,ix=int(x*stride)+kx*int(dilation);
            if(iy>=0 && iy<int(h) && ix>=0 && ix<int(w))sum+=int(channels)*input[iy*w+ix];
        }
        output[y*ow+x]=std::clamp(sum>>shift,0,limit);
    }
    return output;
}
Plane transpose(const Plane& input,unsigned w,unsigned h,unsigned channels) {
    Plane output(4*w*h);
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)
        for(int ky=-1;ky<=1;++ky)for(int kx=-1;kx<=1;++kx) {
            const int oy=int(2*y)+ky,ox=int(2*x)+kx;
            if(oy>=0 && oy<int(2*h) && ox>=0 && ox<int(2*w))
                output[oy*2*w+ox]+=int(channels)*input[y*w+x];
        }
    for(auto& value:output)value=std::clamp(value>>8,0,255);
    return output;
}
Plane nonzero_reference(unsigned w,unsigned h) {
    if(w<8 || h<8 || w%8 || h%8)throw std::runtime_error("Reference dimensions must be multiples of eight");
    // Warm RGB=(255,239,223): XOR codes=(127,111,95), mean111.
    // All-one first-layer weights make their sum equal to three111 channels.
    Plane encoded(w*h,111);
    encoded=convolution(encoded,w,h,3,2,1,0);
    encoded=convolution(encoded,w/2,h/2,8,2,1,0);
    encoded=convolution(encoded,w/4,h/4,16,2,1,0);
    for(auto& value:encoded)value*=32;
    Plane shrink(encoded.size());
    for(unsigned dilation=1;dilation<=4;++dilation) {
        const auto branch=convolution(encoded,w/8,h/8,1,1,dilation,8);
        for(unsigned i=0;i<branch.size();++i)shrink[i]+=64*branch[i];
    }
    for(auto& value:shrink)value=std::clamp(value>>8,0,255);
    auto decoded=transpose(shrink,w/8,h/8,64);
    decoded=transpose(decoded,w/4,h/4,32);
    decoded=transpose(decoded,w/2,h/2,16);
    return convolution(decoded,w,h,8,1,1,0,16384);
}
uint32_t expected_rgb(int mask) {
    const int previous[]={255,239,223};
    uint32_t rgb=0;
    for(unsigned ch=0;ch<3;++ch) {
        const unsigned color=(previous[ch]*(16384+mask)+128*(16384-mask)+16384)>>15;
        rgb|=color<<(8*ch);
    }
    return rgb;
}
}
