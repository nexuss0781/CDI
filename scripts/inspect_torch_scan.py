import inspect
import torch
from torch._higher_order_ops import scan

print("signature", inspect.signature(scan))
print("doc", inspect.getdoc(scan))
print("module", scan)
