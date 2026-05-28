#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/FusedAdagrad.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_adagrad.h>
#include <ATen/ops/_fused_adagrad_native.h>
#endif

namespace at::native {

// ---------------------------------------------------------------------------
// Per-element Adagrad SYCL kernel functor
//
// Math (matches CPU/CUDA implementation):
//   if grad_scale:  grad /= grad_scale
//   if maximize:    grad = -grad
//   if weight_decay: grad += param * weight_decay
//   state_sum += grad * grad
//   param -= clr * grad / (sqrt(state_sum) + eps)
// ---------------------------------------------------------------------------
template <typename scalar_t>
struct FusedAdagradKernelFunctor {
  using opmath_t = at::opmath_type<scalar_t>;

  scalar_t* param_ptr;
  scalar_t* grad_ptr;
  scalar_t* state_sum_ptr;
  double clr;
  double eps;
  double weight_decay;
  bool maximize;
  const float* grad_scale_ptr;

  void operator()(sycl::item<1> item) const {
    const int64_t i = static_cast<int64_t>(item.get_id(0));

    opmath_t param = static_cast<opmath_t>(param_ptr[i]);
    opmath_t grad = static_cast<opmath_t>(grad_ptr[i]);
    opmath_t state_sum = static_cast<opmath_t>(state_sum_ptr[i]);

    if (grad_scale_ptr) {
      grad /= static_cast<opmath_t>(*grad_scale_ptr);
      // store unscaled grad back (matches CUDA behavior)
      grad_ptr[i] = static_cast<scalar_t>(grad);
    }
    if (maximize) {
      grad = -grad;
    }
    if (weight_decay != 0.0) {
      grad += param * static_cast<opmath_t>(weight_decay);
    }

    state_sum += grad * grad;
    state_sum_ptr[i] = static_cast<scalar_t>(state_sum);

    param -= static_cast<opmath_t>(clr) * grad /
        (sycl::sqrt(state_sum) + static_cast<opmath_t>(eps));
    param_ptr[i] = static_cast<scalar_t>(param);
  }
};

// ---------------------------------------------------------------------------
// Per-tensor dispatch: compute corrected_lr, then submit SYCL kernel
// ---------------------------------------------------------------------------
static void fused_adagrad_kernel_xpu(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& state_sum,
    const at::Tensor& state_step,
    const double lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const float* grad_scale_ptr) {
  const double step = static_cast<double>(state_step.item<float>());
  const double clr = lr / (1.0 + (step - 1.0) * lr_decay);
  const int64_t n = param.numel();
  if (n == 0) {
    return;
  }

  sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16,
      kHalf,
      param.scalar_type(),
      "fused_adagrad_kernel_xpu",
      [&]() {
        using scalar_t = scalar_t;
        FusedAdagradKernelFunctor<scalar_t> functor{
            param.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            state_sum.data_ptr<scalar_t>(),
            clr,
            eps,
            weight_decay,
            maximize,
            grad_scale_ptr};
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for<FusedAdagradKernelFunctor<scalar_t>>(
              sycl::range<1>(static_cast<size_t>(n)), functor);
        });
      });
}

// ---------------------------------------------------------------------------
// XPU entry point: float lr overload
// Mirrors _fused_adagrad_kernel_cpu_ in FusedAdagrad.cpp
// ---------------------------------------------------------------------------
void _fused_adagrad_kernel_xpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    const double lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;

  // If found_inf is set (e.g. AMP overflow), skip the update entirely
  if (found_inf_ptr) {
    // found_inf tensor lives on XPU; bring the scalar to CPU to check
    float found_inf_val =
        found_inf->to(at::kCPU).item<float>();
    if (found_inf_val == 1.0f) {
      return;
    }
  }

  const size_t n_tensors = params.size();
  TORCH_CHECK(grads.size() == n_tensors);
  TORCH_CHECK(state_sums.size() == n_tensors);
  TORCH_CHECK(state_steps.size() == n_tensors);

  for (size_t i = 0; i < n_tensors; i++) {
    fused_adagrad_stub(
        kXPU,
        params[i],
        grads[i],
        state_sums[i],
        state_steps[i],
        lr,
        lr_decay,
        weight_decay,
        eps,
        maximize,
        grad_scale_ptr);
  }
}

// ---------------------------------------------------------------------------
// XPU entry point: Tensor lr overload
// Mirrors CUDA behavior: if lr is on CPU convert to scalar;
// if lr is on XPU validate device then use scalar value.
// ---------------------------------------------------------------------------
void _fused_adagrad_kernel_xpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (lr.is_cpu()) {
    _fused_adagrad_kernel_xpu_(
        params,
        grads,
        state_sums,
        state_steps,
        lr.item<double>(),
        lr_decay,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
    return;
  }

  // lr is on XPU — validate it lives on the same device as params
  const Device param_device = params[0].device();
  TORCH_CHECK(
      lr.device() == param_device,
      "lr must be on the same XPU device as the params");
  if (grad_scale.has_value()) {
    TORCH_CHECK(
        grad_scale->device() == param_device,
        "grad_scale must be on the same XPU device as the params");
  }
  if (found_inf.has_value()) {
    TORCH_CHECK(
        found_inf->device() == param_device,
        "found_inf must be on the same XPU device as the params");
  }

  _fused_adagrad_kernel_xpu_(
      params,
      grads,
      state_sums,
      state_steps,
      lr.item<double>(),
      lr_decay,
      weight_decay,
      eps,
      maximize,
      grad_scale,
      found_inf);
}

// Register XPU dispatch for the fused_adagrad_stub
REGISTER_XPU_DISPATCH(fused_adagrad_stub, &fused_adagrad_kernel_xpu)

} // namespace at::native
