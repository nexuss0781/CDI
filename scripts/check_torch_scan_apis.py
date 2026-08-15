import torch

print({
    "torch_version": torch.__version__,
    "has_func_scan": hasattr(torch.func, "scan"),
    "has_associative_scan": hasattr(torch, "associative_scan"),
    "higher_order_ops": sorted(name for name in dir(torch._higher_order_ops) if "scan" in name.lower() or "while" in name.lower()),
})
