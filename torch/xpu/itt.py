import torch
import torch._C._XPU._itt as _C_XPU_ITT

# 这是一个 "哑" 类型，只用于类型提示
class IttDeviceRangeHandle:
    pass

def device_range_start(msg: str, stream: "torch.xpu.Stream" = None) -> IttDeviceRangeHandle:
    """
    Starts a device-synchronized ITT task range on a specific XPU stream.
    """
    if not torch.xpu.is_available():
        raise RuntimeError("torch.xpu.itt requires XPU")
    
    if stream is None:
        stream = torch.xpu.current_stream()

    return _C_XPU_ITT.device_range_start(stream.sycl_queue, msg)

def device_range_end(handle: IttDeviceRangeHandle, stream: "torch.xpu.Stream" = None):
    """
    Ends a device-synchronized ITT task range.
    """
    if not torch.xpu.is_available():
        raise RuntimeError("torch.xpu.itt requires XPU")

    if stream is None:
        stream = torch.xpu.current_stream()
        
    _C_XPU_ITT.device_range_end(stream.sycl_queue, handle)