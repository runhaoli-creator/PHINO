#!/usr/bin/env python3
"""Measure inference latency for all backbones."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch, time

device = 'cuda:0'
dummy = torch.randn(64, 3, 224, 224, device=device)

def measure(model, name, n_warmup=10, n_measure=50):
    for _ in range(n_warmup):
        with torch.no_grad():
            model(dummy)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    avg_ms = sum(times) / len(times) * 1000
    img_per_sec = 64 / (sum(times) / len(times))
    per_img_ms = avg_ms / 64
    print(f'{name}: {per_img_ms:.1f} ms/img, {img_per_sec:.0f} img/s')

# Voltron
from backbone_utils import load_voltron, load_theia
voltron_modelload_voltron(device)
measure(voltron_model, 'Voltron')
del voltron_model
torch.cuda.empty_cache()

# VC-1
import vc_models
from vc_models.models.vit import model_utils as vc1_utils
vc1_model, _, _, _ = vc1_utils.load_model(vc1_utils.VC1_BASE_NAME)
vc1_model = vc1_model.to(device).eval()
measure(vc1_model, 'VC-1')
del vc1_model
torch.cuda.empty_cache()

# Theia
theia_modeld_theia(device)
measure(theia_model, 'Theia')
del theia_model
torch.cuda.empty_cache()

# MVP
import mvp
mvp_model = mvp.load('vitb-mae-egosoup')
mvp_model = mvp_model.to(device).eval()
measure(mvp_model, 'MVP')
del mvp_model
torch.cuda.empty_cache()
