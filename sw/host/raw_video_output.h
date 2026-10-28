#pragma once
#include <ostream>

namespace fiaccel {
// Keep a private handle to the original binary stdout pipe. XRT and its
// simulator children may write diagnostics to stdout, which goes to stderr
// after construction. Only this stream writes pixels to the original pipe.
class RawVideoOutput {
    class Buffer:public std::streambuf {
        int fd_;
        std::streamsize xsputn(const char* data,std::streamsize count) override;
        int_type overflow(int_type value) override;
        int sync() override {return 0;}
    public:
        Buffer();
        ~Buffer() override;
    } buffer_;
    std::ostream stream_;
public:
    RawVideoOutput():stream_(&buffer_) {}
    RawVideoOutput(const RawVideoOutput&)=delete;
    RawVideoOutput& operator=(const RawVideoOutput&)=delete;
    std::ostream& stream() {return stream_;}
};
}
