#pragma once
#include <cstdint>
#include <vector>
namespace fiaccel {
std::vector<int> nonzero_reference(unsigned width,unsigned height);
uint32_t expected_rgb(int mask);
}
