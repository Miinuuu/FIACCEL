#include "raw_video_output.h"
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <system_error>
#include <unistd.h>

namespace fiaccel {
RawVideoOutput::Buffer::Buffer() {
    std::cout.flush();std::fflush(stdout);
    fd_=fcntl(STDOUT_FILENO,F_DUPFD_CLOEXEC,3);
    if(fd_<0)throw std::system_error(errno,std::generic_category(),"Duplicate video output pipe");
    if(dup2(STDERR_FILENO,STDOUT_FILENO)<0) {
        const int error=errno;close(fd_);
        throw std::system_error(error,std::generic_category(),"Redirect runtime diagnostics");
    }
}
RawVideoOutput::Buffer::~Buffer() {close(fd_);}
std::streamsize RawVideoOutput::Buffer::xsputn(const char* data,std::streamsize count) {
    std::streamsize done=0;
    while(done<count) {
        const auto bytes=write(fd_,data+done,static_cast<std::size_t>(count-done));
        if(bytes<0 && errno==EINTR)continue;
        if(bytes<=0)break;
        done+=bytes;
    }
    return done;
}
RawVideoOutput::Buffer::int_type RawVideoOutput::Buffer::overflow(int_type value) {
    if(traits_type::eq_int_type(value,traits_type::eof()))return traits_type::not_eof(value);
    const char byte=traits_type::to_char_type(value);
    return xsputn(&byte,1)==1 ? value : traits_type::eof();
}
}
