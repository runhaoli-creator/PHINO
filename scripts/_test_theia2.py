"""Test Theia loading with transformers 5.x patches."""
import transformers.modeling_utils as mu
import transformers.integrations.accelerate as acc_module
import contextlib

# Patch 1: Fix meta device context error for inner from_pretrained calls
# Must patch in BOTH the source module AND modeling_utils where it's imported
acc_module.check_and_set_device_map = lambda x: x
mu.check_and_set_device_map = lambda x: x

# Patch 2: Disable fast init (meta device) context 
mu._fast_init_context = lambda *a, **kw: contextlib.nullcontext()

# Patch 3: Fix missing all_tied_weights_keys attribute
orig_mark = mu.PreTrainedModel.mark_tied_weights_as_initialized
def patched_mark(self, loading_info):
    if not hasattr(self, 'all_tied_weights_keys'):
        self.all_tied_weights_keys = {}
    return orig_mark(self, loading_info)
mu.PreTrainedModel.mark_tied_weights_as_initialized = patched_mark

from transformers import AutoModel
import torch

print('Loading Theia...')
model = AutoModel.from_pretrained(
    'theaiinstitute/theia-base-patch16-224-cdiv', 
    trust_remote_code=True
)
model.eval()
print(f'Model type: {type(model)}')
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

# Test forward pass
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model(x)

if isinstance(out, dict):
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f'{k}: {v.shape}')
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            print(f'{k}: list of {len(v)} tensors, first: {v[0].shape}')
elif hasattr(out, 'last_hidden_state'):
    print(f'last_hidden_state: {out.last_hidden_state.shape}')
else:
    print(f'Output type: {type(out)}')
    
print('Theia OK!')
