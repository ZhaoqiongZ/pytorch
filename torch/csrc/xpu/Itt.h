#pragma once
#include <torch/csrc/utils/pybind.h>
#include <cstdint> 
#include <string>
#include <torch/csrc/itt_wrapper.h> 

namespace torch::xpu::itt {

using torch::profiler::itt_handle_t;


struct IttDeviceRangeHandle {
  std::string msg;
  itt_handle_t task_id = 0; 

  IttDeviceRangeHandle(const char* m) : msg(m) {}
};

void initXPUITTBindings(PyObject* module);

} // namespace torch::xpu::itt