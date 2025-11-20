#include <torch/csrc/xpu/Itt.h>
#include <sycl/sycl.hpp> // [XPU 依赖] 包含 SYCL API
#include <iostream> // 调试用
#include <unistd.h> 
#include <sys/syscall.h>
#include <thread> // 用于 std::this_thread::sleep_for
#include <chrono> // 用于 std::chrono::microseconds

#include <torch/csrc/xpu/Stream.h>

// [关键] 包含你创建的 "CPU ITT 工具箱" 的实现
#include <torch/csrc/itt_wrapper.h>

namespace torch::xpu::itt {

// 使用你创建的 CPU 函数
using torch::profiler::itt_handle_t;
using torch::profiler::itt_range_start;
using torch::profiler::itt_range_end;

// --- 回调函数 (Callbacks) ---
// (这些在 CPU 上运行，但与 GPU 流同步)

static void itt_callback_range_start(IttDeviceRangeHandle* handle) {
  // [调用 CPU 工具箱]
  handle->task_id = itt_range_start(handle->msg.c_str());
}

static void itt_callback_range_end(IttDeviceRangeHandle* handle) {
  // [调用 CPU 工具箱]
  itt_range_end(handle->task_id);
  delete handle; // 清理内存
}


// --- 2. 修改“启动”和“结束”函数的实现 ---
// --- "启动" 函数 (修改返回值) ---
static void* itt_device_range_start( // <-- [修改] 返回 void*
    intptr_t queue_ptr, 
    const char* msg) {
  



  sycl::queue* q = reinterpret_cast<sycl::queue*>(queue_ptr);
  IttDeviceRangeHandle* handle = new IttDeviceRangeHandle(msg);


    // 1. 获取当前 OS 线程 ID
  long tid = syscall(SYS_gettid); 
  
  // 2. [关键] 打印 TID。使用 std::cerr 避免缓冲
  std::cerr << "[ITT_DEBUG] START callback TID: " << tid << " Task: " << handle->msg << std::endl;


  q->submit([&](sycl::handler& cgh) {
    cgh.ext_oneapi_barrier();
  });
  
  q->submit([&](sycl::handler& cgh) {
    cgh.host_task([=]() { itt_callback_range_start(handle); });
  });

  //std::this_thread::sleep_for(std::chrono::microseconds(5000000));
  usleep(5000);
  // [修改] C-style cast to void*
  return (void*)handle; 
}

// --- "结束" 函数 (修改参数类型) ---
static void itt_device_range_end(
    intptr_t queue_ptr,
    void* handle_ptr) { // <-- [修改] 接受 void*



  sycl::queue* q = reinterpret_cast<sycl::queue*>(queue_ptr);
  
  // [修改] Cast it back to the type we need for the callback
  IttDeviceRangeHandle* handle = (IttDeviceRangeHandle*)handle_ptr; 

  // 1. 获取当前 OS 线程 ID
  long tid = syscall(SYS_gettid); 
  
  // 2. [关键] 打印 TID
  std::cerr << "[ITT_DEBUG] END callback TID: " << tid << " Task: " << handle->msg << std::endl;

  q->submit([&](sycl::handler& cgh) {
    cgh.host_task([=]() { itt_callback_range_end(handle); });
  });
}
// --- Python 桥梁 (Pybind11) ---

void initXPUITTBindings(PyObject* module) {
  // 'module' 现在是 _XPU 子模块
  auto m =
      py::handle(module).cast<py::module>().def_submodule("_itt", "XPU ITT Bindings");



  m.def(
      "device_range_start",
      &itt_device_range_start,
      py::arg("queue_ptr"),
      py::arg("msg"),
      "Starts a device-synchronized ITT task range.");

  m.def(
      "device_range_end",
      &itt_device_range_end,
      py::arg("queue_ptr"),
      py::arg("handle"),
      "Ends a device-synchronized ITT task range.");
}

} // namespace torch::xpu::itt