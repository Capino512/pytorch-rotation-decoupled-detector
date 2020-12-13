# xtorch

This is a simple encapsulation of pytorch, so that in_featuers/in_channels can be implicitly determined when the model is defined, rather than explicitly specified.

A simple example is as follows:

```python
import torch
from torch import nn
from xtorch import xnn

model = xnn.Sequential(xnn.Linear(16), nn.ReLU(), xnn.Linear(2))
# <===> nn.Sequential(nn.Linear(8, 16), nn.ReLU(), xnn.Linear(16, 2))

model.build_pipe(shape=[2, 8])
 # alternative 
 # model.build(torch.randn(2, 8))

x = torch.randn(32, 8)
y = model(x)
```
