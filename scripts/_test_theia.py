#!/usr/bin/env python3
"""Test Theia model loading."""
import torch
import transformers.integrations.accelerate as acc_module

# Patch to fix meta device error in transformers 5.x
acc_module.check_and_set_device_map = lambda x: x

import transformers.modeling_utils as mu
import contextlib
if hasattr(mu, '_fast_init_context'):
    mu._fast_init_context = lambda *a, **kw: contextlib.nullcontext()

from transformers import AutoModel

print("Loading Theia...")
model = AutoModel.from_pretrained(
    'theaiinstitute/theia-base-patch16-224-cdiv',
    trust_remote_code=True,
    low_cpu_mem_usage=False
)
model.eval()
print(f'Theia loaded: {type(model).__name__}')
total = sum(p.numel() for p in model.parameters())
print(f'Params: {total/1e6:.1f}M')

dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model(dummy)

print(f'Output type: {type(out)}')
if hasattr(out, 'keys'):
    for k in out.keys():
        v = out[k]
        if isinstance(v, torch.Tensor):
            print(f'  {k}: {v.shape}')
elif hasattr(out, 'last_hidden_state'):
    print(f'  last_hidden_state: {out.last_hidden_state.shape}')
elif isinstance(out, torch.Tensor):
    print(f'  tensor: {out.shape}')
else:
    print(f'  value: {out}')
