#ifndef PROFILER_ITT_H
#define PROFILER_ITT_H
#include <c10/macros/Export.h>
#include <cstdint>


namespace torch::profiler {
using itt_handle_t = std::uint64_t;
TORCH_API bool itt_is_available();
TORCH_API void itt_range_push(const char* msg);
TORCH_API void itt_range_pop();
TORCH_API void itt_mark(const char* msg);

TORCH_API itt_handle_t itt_range_start(const char* msg);
TORCH_API void itt_range_end(itt_handle_t handle);
} // namespace torch::profiler

#endif // PROFILER_ITT_H
