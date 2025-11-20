#include <ittnotify.h>
#include <torch/csrc/itt_wrapper.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <atomic>

namespace torch::profiler {
static __itt_domain* _itt_domain = __itt_domain_create("PyTorch");
static std::atomic<itt_handle_t> g_itt_task_id{1};

static std::atomic<int> g_pop_counter{0};

bool itt_is_available() {
  return torch::profiler::impl::ittStubs()->enabled();
}

void itt_range_push(const char* msg) {
  __itt_string_handle* hsMsg = __itt_string_handle_create(msg);
  __itt_task_begin(_itt_domain, __itt_null, __itt_null, hsMsg);
}

void itt_range_pop() {
  __itt_task_end(_itt_domain);
}

void itt_mark(const char* msg) {
  __itt_string_handle* hsMsg = __itt_string_handle_create(msg);
  __itt_task_begin(_itt_domain, __itt_null, __itt_null, hsMsg);
  __itt_task_end(_itt_domain);
}


itt_handle_t itt_range_start(const char* msg) {
  __itt_string_handle* hsMsg = __itt_string_handle_create(msg);
  itt_handle_t task_id = g_itt_task_id.fetch_add(1, std::memory_order_relaxed);
  __itt_id itt_id = {task_id, 0, 0};
  __itt_task_begin_overlapped(_itt_domain, itt_id, __itt_null, hsMsg);
  return task_id;
  //__itt_task_begin(_itt_domain, __itt_null, __itt_null, hsMsg);
}

void itt_range_end(itt_handle_t task_id) {
  __itt_id itt_id = {task_id, 0, 0};
  //__itt_task_end(_itt_domain);
  __itt_task_end_overlapped(_itt_domain, itt_id);

  g_pop_counter.fetch_add(1);
}


int itt_get_pop_count() {
  return g_pop_counter.load();
}


// [新增] 实现这个线程命名函数
void itt_thread_set_name(const char* name) {
  __itt_thread_set_name(name);
}

} // namespace torch::profiler
