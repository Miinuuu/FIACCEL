ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
OUT ?= $(ROOT)/build/u200
XRT ?= /opt/xilinx/xrt
CXX ?= g++
PYTHON ?= python3
SOURCES := $(wildcard $(ROOT)/sw/host/*.cpp)
HEADERS := $(wildcard $(ROOT)/sw/host/*.h)

.PHONY: host test
test:
	cd $(ROOT) && $(PYTHON) -m unittest sw.verification.test_host_pipeline sw.verification.test_u200_model -v
host: $(OUT)/host

$(OUT)/host: $(SOURCES) $(HEADERS)
	mkdir -p $(OUT)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -I$(XRT)/include $(SOURCES) \
		-L$(XRT)/lib -Wl,-rpath,$(XRT)/lib -lxrt_coreutil -pthread -o $@
