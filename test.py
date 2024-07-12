from archs.scratch import DiT_models
#from archs.dinext import DiT_models
#from archs.dit import DiT_models

import torch

x = torch.zeros((1, 4, 256, 256)).cuda().float()
t = torch.zeros((1,)).cuda().long()
y = torch.zeros((1,)).cuda().long()
m = DiT_models["DiT-S/2"]().cuda()
y = m(x, t, y)
print(y.shape)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(m))

from torchstat import stat

stat(m.cpu(), (4, 256, 256))