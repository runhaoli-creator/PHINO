import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import json, sys, torch, numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
sys.path.insert(0, ".")

device = torch.device("cuda:0")

from vc_models.models.vit import model_utils as vc_mu
vc1_model, _, vc1_transform, _ = vc_mu.load_model(vc_mu.VC1_BASE_NAME)
vc1_model = vc1_model.eval().to(device)
print("Loaded VC-1")

from dynaclip.data.dataset import PhysicsProbeDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T

transform = T.Compose([
    T.Resize(224), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = PhysicsProbeDataset("data_cache/dynaclip_data", split="train", transform=transform)
test_ds = PhysicsProbeDataset("data_cache/dynaclip_data", split="test", transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)
print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

def extract_features(loader, model, dev):
    feats, masses, frictions, rests, cats = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(dev)
            out = model(imgs)
            if isinstance(out, dict):
                f = out.get("cls_token", out.get("last_hidden_state", next(iter(out.values()))))
                if f.dim() == 3: f = f[:, 0]
            elif out.dim() == 3: f = out[:, 0]
            else: f = out
            feats.append(f.cpu())
            masses.append(batch["mass"])
            frictions.append(batch["static_friction"])
            rests.append(batch["restitution"])
            cats.extend(batch["category"])
    return (torch.cat(feats).numpy(), torch.cat(masses).numpy(),
            torch.cat(frictions).numpy(), torch.cat(rests).numpy(), cats)

print("Extracting train features...")
tr_f, tr_m, tr_fr, tr_r, tr_c = extract_features(train_loader, vc1_model, device)
print(f"Train features: {tr_f.shape}")
print("Extracting test features...")
te_f, te_m, te_fr, te_r, te_c = extract_features(test_loader, vc1_model, device)
print(f"Test features: {te_f.shape}")

results = {}
print("\n=== LINEAR PROBING ===")
for prop, tr_y, te_y in [("mass", tr_m, te_m), ("friction", tr_fr, te_fr), ("restitution", tr_r, te_r)]:
    reg = Ridge(alpha=1.0); reg.fit(tr_f, tr_y)
    r2 = r2_score(te_y, reg.predict(te_f))
    print(f"  {prop}: R2 = {r2:.4f}")
    results[f"linear_{prop}_r2"] = float(r2)

le = LabelEncoder()
tr_c_enc = le.fit_transform(tr_c); te_c_enc = le.transform(te_c)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000, C=1.0); clf.fit(tr_f, tr_c_enc)
cat_acc = accuracy_score(te_c_enc, clf.predict(te_f))
print(f"  category: acc = {cat_acc:.4f}")
results["linear_category_acc"] = float(cat_acc)

print("\n=== k-NN (k=5) ===")
from sklearn.neighbors import NearestNeighbors
nn_model = NearestNeighbors(n_neighbors=5, metric="cosine"); nn_model.fit(tr_f)
dists, indices = nn_model.kneighbors(te_f)
weights = 1.0 / (dists + 1e-8); weights /= weights.sum(axis=1, keepdims=True)
for prop, tr_y, te_y in [("mass", tr_m, te_m), ("friction", tr_fr, te_fr), ("restitution", tr_r, te_r)]:
    r2 = r2_score(te_y, (weights * tr_y[indices]).sum(axis=1))
    print(f"  {prop}: R2 = {r2:.4f}")
    results[f"knn_{prop}_r2"] = float(r2)

print("\n=== CLUSTERING ===")
all_f = np.concatenate([tr_f, te_f])
all_c = np.array(LabelEncoder().fit_transform(tr_c + te_c))
n_clusters = len(np.unique(all_c))
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
labels = kmeans.fit_predict(all_f)
nmi = normalized_mutual_info_score(all_c, labels)
ari = adjusted_rand_score(all_c, labels)
print(f"  NMI = {nmi:.4f}, ARI = {ari:.4f}")
results["clustering_nmi"] = float(nmi)
results["clustering_ari"] = float(ari)

out_path = Path("results/physics_v2/vc1_results.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f: json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
print(json.dumps(results, indent=2))
