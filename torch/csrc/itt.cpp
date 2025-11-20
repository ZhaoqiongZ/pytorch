#include <torch/csrc/itt.h>
#include <torch/csrc/itt_wrapper.h>

namespace torch::profiler {
void initIttBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto itt = m.def_submodule("_itt", "VTune ITT bindings");
  itt.def("is_available", itt_is_available);
  itt.def("rangePush", itt_range_push);
  itt.def("rangePop", itt_range_pop);
  itt.def("mark", itt_mark);
  itt.def("rangeStart", itt_range_start, py::arg("msg"));
  itt.def("rangeEnd", itt_range_end, py::arg("handle"));
  itt.def("thread_set_name", itt_thread_set_name, py::arg("name"),"Sets the name of the current thread or a new thread for VTune to track."); 
  itt.def("get_pop_count",itt_get_pop_count,"Returns the number of ITT tasks successfully popped by the C++ implementation.");
}
} // namespace torch::profiler
