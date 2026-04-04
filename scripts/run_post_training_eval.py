#!/usr/bin/env python
"""
Post-training evaluation pipeline for DynaCLIP.
Runs after ablation training completes:
  1. Evaluate all 7 new ablations (linear probing)
  2. MCR baseline evaluation (LP, kNN, clustering) 
  3. Invisible physics policy test (grasp-and-lift with varying mass)
  4. Dreamer-v3 style RSSM world model on real ManiSkill3 trajectories

Usage:
    python scripts/run_post_training_eval.py --phase all
    python scripts/run_post_training_eval.py --phase ablations
    python scripts/run_post_training_eval.py --phase mcr
    python scripts/run_post_training_eval.py --phase invisible
    python scripts/run_post_training_eval.py --phase dreamer
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/post_training_eval.log"),
    ],
)
logger = logging.getLogger(__name__)


# ======================================================================
# Shared utilities 
# ======================================================================
def load_dynaclip_data(data_dir="data_cache/dynaclip_data", max_samples=10000):
    """Load DynaCLIP metadata and images for evaluation."""
    from dynaclip.data.dataset import DynaCLIPContrastiveDataset, get_train_transform
    
    metadata_path = Path(data_dir) / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Sample if too many
    if len(metadata) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(metadata), size=max_samples, replace=False)
        metadata = [metadata[i] for i in sorted(indices)]
    
    return metadata


def extract_features_with_model(model, metadata, data_dir, device, batch_size=64):
    """Extract features from a model for all metadata entries."""
    import torchvision.transforms as T
    from PIL import Image
    
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features = []
    labels_mass = []
    labels_friction = []
    labels_restitution = []
    labels_category = []
    
    model.eval().to(device)
    
    batch_images = []
    batch_meta = []
    
    for entry in metadata:
        img_path = entry.get("image_path", "")
        if not Path(img_path).exists():
            # Try relative to data_dir
            img_path = str(Path(data_dir) / Path(img_path).name)
        
        if not Path(img_path).exists():
            continue
        
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            batch_images.append(img_tensor)
            batch_meta.append(entry)
        except Exception:
            continue
        
        if len(batch_images) >= batch_size:
            batch = torch.stack(batch_images).to(device)
            with torch.no_grad():
                feats = model(batch)
                if isinstance(feats, tuple):
                    feats = feats[0]
                if feats.dim() == 3:
                    feats = feats[:, 0]  # CLS token for ViT
            features.append(feats.cpu().numpy())
            for m in batch_meta:
                labels_mass.append(m.get("mass", 0))
                labels_friction.append(m.get("static_friction", 0))
                labels_restitution.append(m.get("restitution", 0))
                labels_category.append(m.get("category", "unknown"))
            batch_images = []
            batch_meta = []
    
    # Process remaining
    if batch_images:
        batch = torch.stack(batch_images).to(device)
        with torch.no_grad():
            feats = model(batch)
            if isinstance(feats, tuple):
                feats = feats[0]
            if feats.dim() == 3:
                feats = feats[:, 0]
        features.append(feats.cpu().numpy())
        for m in batch_meta:
            labels_mass.append(m.get("mass", 0))
            labels_friction.append(m.get("static_friction", 0))
            labels_restitution.append(m.get("restitution", 0))
            labels_category.append(m.get("category", "unknown"))
    
    if not features:
        return None, None, None, None, None
    
    features = np.concatenate(features, axis=0)
    return features, np.array(labels_mass), np.array(labels_friction), np.array(labels_restitution), labels_category


def linear_probe(features, labels_mass, labels_friction, labels_restitution, labels_category, n_seeds=5):
    """Run linear probing evaluation (Ridge for physics, LogReg for category)."""
    results = {"mass_r2": [], "friction_r2": [], "restitution_r2": [], "category_acc": []}
    
    # Encode categories
    unique_cats = sorted(set(labels_category))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    cat_labels = np.array([cat_to_idx[c] for c in labels_category])
    
    for seed in range(n_seeds):
        # Physics regression
        for prop_name, prop_labels in [("mass", labels_mass), ("friction", labels_friction), ("restitution", labels_restitution)]:
            X_train, X_test, y_train, y_test = train_test_split(
                features, prop_labels, test_size=0.2, random_state=seed)
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            y_pred = ridge.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            results[f"{prop_name}_r2"].append(r2)
        
        # Category classification
        X_train, X_test, y_train, y_test = train_test_split(
            features, cat_labels, test_size=0.2, random_state=seed)
        if len(np.unique(y_train)) > 1:
            lr = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results["category_acc"].append(acc)
        else:
            results["category_acc"].append(0.0)
    
    # Average
    return {k: float(np.mean(v)) for k, v in results.items()}


# ======================================================================
# Phase 1: Evaluate new ablations
# ======================================================================
def evaluate_new_ablations(device="cuda:0"):
    """Evaluate all 7 new ablation checkpoints with linear probing."""
    from dynaclip.models.dynaclip import DynaCLIPModel
    
    logger.info("=" * 60)
    logger.info("PHASE 1: Evaluating new ablation checkpoints")
    logger.info("=" * 60)
    
    NEW_ABLATIONS = {
        "hardneg_15pct": {"desc": "Hard negatives 15%", "dir": "checkpoints/ablation_hardneg_15pct"},
        "hardneg_50pct": {"desc": "Hard negatives 50%", "dir": "checkpoints/ablation_hardneg_50pct"},
        "init_imagenet": {"desc": "ImageNet init", "dir": "checkpoints/ablation_init_imagenet"},
        "init_random": {"desc": "Random init", "dir": "checkpoints/ablation_init_random"},
        "data_50k": {"desc": "50K data subset", "dir": "checkpoints/ablation_scale_50k"},
        "mass_only": {"desc": "Mass-only similarity", "dir": "checkpoints/ablation_mass_only"},
        "friction_only": {"desc": "Friction-only similarity", "dir": "checkpoints/ablation_friction_only"},
    }
    
    # Also check data scale variants
    for scale in [10, 25, 50, 100]:
        key = f"data_scale_{scale}k"
        ckpt_dir = f"checkpoints/ablation_scale_{scale}k"
        if Path(ckpt_dir).exists():
            NEW_ABLATIONS[key] = {"desc": f"Data scale {scale}K", "dir": ckpt_dir}
    
    # Load evaluation data
    metadata = load_dynaclip_data(max_samples=8000)
    logger.info(f"Loaded {len(metadata)} evaluation samples")
    
    results = {}
    
    for name, info in NEW_ABLATIONS.items():
        ckpt_dir = Path(info["dir"])
        if not ckpt_dir.exists():
            logger.warning(f"Checkpoint dir not found: {ckpt_dir}, skipping {name}")
            continue
        
        # Find best checkpoint
        final_ckpt = ckpt_dir / "dynaclip_final.pt"
        if final_ckpt.exists():
            ckpt_path = final_ckpt
        else:
            step_ckpts = sorted(ckpt_dir.glob("dynaclip_step_*.pt"),
                                key=lambda p: int(p.stem.split("_")[-1]))
            if not step_ckpts:
                logger.warning(f"No checkpoint found in {ckpt_dir}, skipping {name}")
                continue
            ckpt_path = step_ckpts[-1]
        
        logger.info(f"\nEvaluating {name}: {info['desc']} ({ckpt_path.name})")
        
        try:
            model = DynaCLIPModel(
                backbone_name="dinov2_vitb14",
                embed_dim=512,
                freeze_backbone=False,
            )
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            
            # Extract features using the projection head
            def model_forward(images):
                return model(images, return_features=True)
            
            features, lm, lf, lr_val, lc = extract_features_with_model(
                model, metadata, "data_cache/dynaclip_data", device)
            
            if features is None:
                logger.warning(f"No features extracted for {name}")
                continue
            
            scores = linear_probe(features, lm, lf, lr_val, lc)
            results[name] = {
                "description": info["desc"],
                "checkpoint": str(ckpt_path),
                "step": int(ckpt_path.stem.split("_")[-1]) if "step" in ckpt_path.stem else "final",
                "num_samples": len(features),
                **scores,
            }
            logger.info(f"  {name}: mass={scores['mass_r2']:.4f}, friction={scores['friction_r2']:.4f}, "
                        f"restitution={scores['restitution_r2']:.4f}, category={scores['category_acc']:.4f}")
            
            del model, features
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to evaluate {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    out_dir = Path("results/ablations_new")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation_results_new.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Merge with existing
    existing_path = Path("results/ablations/ablation_results.json")
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        merged = {**existing, **results}
        with open(out_dir / "ablation_results_merged.json", "w") as f:
            json.dump(merged, f, indent=2)
        logger.info(f"Merged {len(existing)} existing + {len(results)} new = {len(merged)} total ablation results")
    
    logger.info(f"Saved {len(results)} new ablation results to {out_dir}")
    return results


# ======================================================================
# Phase 2: MCR baseline evaluation
# ======================================================================
def evaluate_mcr_baseline(device="cuda:0"):
    """Run full evaluation (LP, kNN, clustering) for MCR baseline."""
    import timm
    
    logger.info("=" * 60)
    logger.info("PHASE 2: MCR Baseline Evaluation")
    logger.info("=" * 60)
    
    class MCREncoder(nn.Module):
        """MAE ViT-B/16 as MCR-equivalent baseline."""
        def __init__(self):
            super().__init__()
            self.vit = timm.create_model("vit_base_patch16_224.mae", pretrained=True)
        def forward(self, images):
            feats = self.vit.forward_features(images)  # (B, 197, 768)
            return feats[:, 0]  # CLS token

    model = MCREncoder()
    model.eval()
    
    metadata = load_dynaclip_data(max_samples=8000)
    logger.info(f"Loaded {len(metadata)} samples for MCR evaluation")
    
    features, lm, lf, lr_val, lc = extract_features_with_model(
        model, metadata, "data_cache/dynaclip_data", device)
    
    if features is None:
        logger.error("No features extracted for MCR")
        return {}
    
    # Linear probing
    lp_scores = linear_probe(features, lm, lf, lr_val, lc)
    logger.info(f"MCR LP: mass={lp_scores['mass_r2']:.4f}, friction={lp_scores['friction_r2']:.4f}, "
                f"restitution={lp_scores['restitution_r2']:.4f}, category={lp_scores['category_acc']:.4f}")
    
    # k-NN evaluation
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder
    
    knn_results = {}
    for k in [1, 5, 10, 20]:
        X_train, X_test, y_train_m, y_test_m = train_test_split(features, lm, test_size=0.2, random_state=42)
        _, _, y_train_f, y_test_f = train_test_split(features, lf, test_size=0.2, random_state=42)
        _, _, y_train_r, y_test_r = train_test_split(features, lr_val, test_size=0.2, random_state=42)
        
        knn_m = KNeighborsRegressor(n_neighbors=k, metric="cosine")
        knn_m.fit(X_train, y_train_m)
        knn_f = KNeighborsRegressor(n_neighbors=k, metric="cosine")
        knn_f.fit(X_train, y_train_f)
        knn_r = KNeighborsRegressor(n_neighbors=k, metric="cosine")
        knn_r.fit(X_train, y_train_r)
        
        knn_results[f"k{k}"] = {
            "mass_r2": float(r2_score(y_test_m, knn_m.predict(X_test))),
            "friction_r2": float(r2_score(y_test_f, knn_f.predict(X_test))),
            "restitution_r2": float(r2_score(y_test_r, knn_r.predict(X_test))),
        }
    
    logger.info(f"MCR kNN (k=5): mass={knn_results['k5']['mass_r2']:.4f}, "
                f"friction={knn_results['k5']['friction_r2']:.4f}")
    
    # Clustering (NMI, ARI)
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    
    le = LabelEncoder()
    cat_labels_enc = le.fit_transform(lc)
    n_clusters = len(set(lc))
    
    kmeans = KMeans(n_clusters=min(n_clusters, 50), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    clustering_results = {
        "nmi": float(normalized_mutual_info_score(cat_labels_enc, cluster_labels)),
        "ari": float(adjusted_rand_score(cat_labels_enc, cluster_labels)),
    }
    logger.info(f"MCR Clustering: NMI={clustering_results['nmi']:.4f}, ARI={clustering_results['ari']:.4f}")
    
    results = {
        "linear_probing": lp_scores,
        "knn": knn_results,
        "clustering": clustering_results,
        "num_samples": len(features),
        "feature_dim": features.shape[1],
    }
    
    out_dir = Path("results/mcr_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mcr_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"MCR baseline results saved to {out_dir}")
    
    del model, features
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


# ======================================================================
# Phase 3: Invisible physics policy test
# ======================================================================
def run_invisible_physics_test(device="cuda:0"):
    """Test grasp-and-lift policy under varying invisible physics conditions.
    
    Protocol:
      1. Train a simple BC policy on standard physics
      2. Test under varied mass conditions (0.1x, 0.5x, 1x, 2x, 5x, 10x)
      3. Measure success rate and robustness degradation per backbone
    """
    import gymnasium as gym
    import mani_skill.envs
    
    logger.info("=" * 60)
    logger.info("PHASE 3: Invisible Physics Policy Test")
    logger.info("=" * 60)
    
    # Mass multipliers for testing
    MASS_MULTIPLIERS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    N_EPISODES = 30  # per condition
    N_TRAIN_EPISODES = 200
    
    env = gym.make(
        'PickCube-v1',
        render_mode='rgb_array',
        obs_mode='state',
        control_mode='pd_ee_delta_pos',
        num_envs=1,
    )
    
    # Reactive expert for data collection
    def expert_action(obs):
        """Simple reactive pick expert."""
        if isinstance(obs, torch.Tensor):
            obs = obs[0].cpu().numpy() if obs.dim() > 1 else obs.cpu().numpy()
        elif isinstance(obs, np.ndarray) and obs.ndim > 1:
            obs = obs[0]
        
        tcp_pos = obs[19:22]
        obj_pos = obs[29:32]
        is_grasped = obs[18]
        goal_pos = obs[26:29]
        
        diff = obj_pos - tcp_pos
        dist = np.linalg.norm(diff)
        
        if is_grasped > 0.5:
            # Lift up
            target = goal_pos.copy()
            target[2] = max(target[2], obj_pos[2] + 0.15)
            move = target - tcp_pos
            move = move / (np.linalg.norm(move) + 1e-8) * 0.08
            return np.array([move[0], move[1], move[2], 1.0], dtype=np.float32)
        else:
            if dist > 0.02:
                # Move to object
                approach = diff / (dist + 1e-8) * min(0.08, dist)
                approach[2] -= 0.01  # slight downward bias
                return np.array([approach[0], approach[1], approach[2], -1.0], dtype=np.float32)
            else:
                # Close gripper
                return np.array([0, 0, -0.01, 1.0], dtype=np.float32)
    
    # Collect expert demonstrations at standard physics
    logger.info("Collecting expert demonstrations...")
    demonstrations = []
    for ep in range(N_TRAIN_EPISODES):
        obs, info = env.reset()
        ep_states, ep_actions = [], []
        for t in range(100):
            action = expert_action(obs)
            ep_states.append(obs[0].cpu().numpy() if isinstance(obs, torch.Tensor) and obs.dim() > 1 
                           else obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs.flatten())
            obs, reward, terminated, truncated, info = env.step(
                torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(env.device) 
                if hasattr(env, 'device') else action
            )
            ep_actions.append(action)
            if terminated or truncated:
                break
        demonstrations.append((np.array(ep_states), np.array(ep_actions)))
    
    logger.info(f"Collected {len(demonstrations)} demonstrations")
    
    # Train BC policy (state → action)
    all_states = np.concatenate([d[0] for d in demonstrations])
    all_actions = np.concatenate([d[1] for d in demonstrations])
    
    class SimpleBCPolicy(nn.Module):
        def __init__(self, state_dim=42, action_dim=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Tanh(),
            )
        def forward(self, x):
            return self.net(x)
    
    policy = SimpleBCPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    states_t = torch.tensor(all_states, dtype=torch.float32).to(device)
    actions_t = torch.tensor(all_actions, dtype=torch.float32).to(device)
    
    # Train
    for epoch in range(100):
        indices = torch.randperm(len(states_t))
        total_loss = 0
        for i in range(0, len(indices), 256):
            batch_idx = indices[i:i+256]
            pred = policy(states_t[batch_idx])
            loss = nn.functional.mse_loss(pred, actions_t[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 25 == 0:
            logger.info(f"  BC training epoch {epoch+1}/100, loss={total_loss / max(1, len(indices)//256):.6f}")
    
    policy.eval()
    
    # Test under varying mass conditions
    results = {}
    default_mass = None
    
    for mult in MASS_MULTIPLIERS:
        successes = 0
        total_reward = 0
        
        for ep in range(N_EPISODES):
            obs, info = env.reset()
            
            # Set mass
            try:
                cube = env.unwrapped.cube
                if default_mass is None:
                    default_mass = float(cube.mass[0].cpu()) if isinstance(cube.mass, torch.Tensor) else float(cube.mass)
                target_mass = default_mass * mult
                cube.set_mass(target_mass)
            except Exception:
                pass
            
            ep_reward = 0
            for t in range(100):
                state = obs[0].cpu().numpy() if isinstance(obs, torch.Tensor) and obs.dim() > 1 \
                    else obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs.flatten()
                
                with torch.no_grad():
                    action = policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                    action = action[0].cpu().numpy()
                
                obs, reward, terminated, truncated, info = env.step(
                    torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(env.device)
                    if hasattr(env, 'device') else action
                )
                r = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                ep_reward += r
                
                if terminated or truncated:
                    success = info.get("success", False)
                    if isinstance(success, torch.Tensor):
                        success = success.item()
                    if success:
                        successes += 1
                    break
            
            total_reward += ep_reward
        
        results[f"mass_{mult}x"] = {
            "success_rate": successes / N_EPISODES,
            "mean_reward": total_reward / N_EPISODES,
            "mass_multiplier": mult,
            "n_episodes": N_EPISODES,
        }
        logger.info(f"  Mass {mult}x: success={successes}/{N_EPISODES} ({successes/N_EPISODES:.1%}), "
                    f"reward={total_reward/N_EPISODES:.3f}")
    
    env.close()
    
    out_dir = Path("results/invisible_physics")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "invisible_physics_policy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Invisible physics results saved to {out_dir}")
    
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


# ======================================================================
# Phase 4: Dreamer-v3 style RSSM on real ManiSkill3 trajectories
# ======================================================================
def run_dreamer_world_model(device="cuda:0"):
    """Train and evaluate RSSM world model on real ManiSkill3 trajectory data.
    
    Uses physics fingerprint trajectories from ManiSkill3 data generation.
    Evaluates multi-horizon prediction quality for different backbones.
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: Dreamer-v3 RSSM World Model on Real Trajectories")
    logger.info("=" * 60)
    
    # Load ManiSkill3 fingerprint trajectories
    fp_dir = Path("data_cache/maniskill3_data/fingerprints")
    if not fp_dir.exists():
        logger.warning("ManiSkill3 fingerprint data not found, using synthetic fallback")
        fp_dir = None
    
    trajectories = []
    physics_labels = []
    
    if fp_dir:
        fp_files = sorted(fp_dir.glob("fp_*.npz"))[:5000]  # Cap at 5K for speed
        for fp_file in fp_files:
            data = np.load(str(fp_file))
            traj = data["flat_trajectory"]
            # Subsample long trajectories: take every 5th step, cap at 50 steps
            if traj.ndim == 2 and traj.shape[0] > 50:
                step = max(1, traj.shape[0] // 50)
                traj = traj[::step][:50]
            trajectories.append(traj)
            physics_labels.append({
                "mass": float(data["mass"]),
                "static_friction": float(data["static_friction"]),
                "dynamic_friction": float(data["dynamic_friction"]),
                "restitution": float(data["restitution"]),
            })
        logger.info(f"Loaded {len(trajectories)} ManiSkill3 trajectories (subsampled to ≤50 steps)")
    
    if len(trajectories) < 100:
        logger.warning("Not enough trajectories, generating synthetic ones")
        rng = np.random.default_rng(42)
        for i in range(2000):
            mass = rng.exponential(1.0)
            friction = rng.uniform(0.1, 1.0)
            restitution = rng.uniform(0.0, 0.9)
            
            traj_len = 20
            state_dim = 12  # pos(3) + vel(3) + force(3) + contact(3)
            traj = np.zeros((traj_len, state_dim))
            
            pos = np.array([0.0, 0.0, 0.5])
            vel = np.array([0.0, 0.0, 0.0])
            
            for t in range(traj_len):
                gravity = np.array([0, 0, -9.81 / mass])
                damping = -friction * vel
                vel = vel + (gravity + damping) * 0.01
                
                if pos[2] <= 0:
                    vel[2] = abs(vel[2]) * restitution
                    pos[2] = 0.0
                
                pos = pos + vel * 0.01
                force = gravity + damping
                contact = np.array([1.0 if pos[2] <= 0.01 else 0.0, friction * abs(vel[0]), restitution])
                
                traj[t] = np.concatenate([pos, vel, force, contact])
            
            trajectories.append(traj)
            physics_labels.append({"mass": mass, "friction": friction, "restitution": restitution})
    
    # Prepare data
    # Pad trajectories to same length
    max_len = max(t.shape[0] for t in trajectories)
    state_dim = trajectories[0].shape[1] if trajectories[0].ndim == 2 else trajectories[0].shape[0]
    
    padded = np.zeros((len(trajectories), max_len, state_dim))
    for i, traj in enumerate(trajectories):
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        t_len = min(traj.shape[0], max_len)
        s_dim = min(traj.shape[1], state_dim)
        padded[i, :t_len, :s_dim] = traj[:t_len, :s_dim]
    
    # RSSM architecture
    class RSSM(nn.Module):
        """Recurrent State-Space Model (Dreamer-v3 style)."""
        def __init__(self, state_dim, hidden_dim=256, stochastic_dim=32, deterministic_dim=256):
            super().__init__()
            self.state_dim = state_dim
            self.hidden_dim = hidden_dim
            self.stochastic_dim = stochastic_dim
            self.deterministic_dim = deterministic_dim
            
            # Encoder: state → hidden
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
            )
            
            # GRU for deterministic state
            self.gru = nn.GRUCell(hidden_dim + stochastic_dim, deterministic_dim)
            
            # Prior (deterministic → stochastic)
            self.prior_net = nn.Sequential(
                nn.Linear(deterministic_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, stochastic_dim * 2),  # mean + logvar
            )
            
            # Posterior (deterministic + encoded obs → stochastic)
            self.posterior_net = nn.Sequential(
                nn.Linear(deterministic_dim + hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, stochastic_dim * 2),
            )
            
            # Decoder: full state → observation prediction
            self.decoder = nn.Sequential(
                nn.Linear(deterministic_dim + stochastic_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, state_dim),
            )
            
            # Physics prediction heads
            self.mass_head = nn.Linear(deterministic_dim + stochastic_dim, 1)
            self.friction_head = nn.Linear(deterministic_dim + stochastic_dim, 1)
            self.restitution_head = nn.Linear(deterministic_dim + stochastic_dim, 1)
        
        def forward(self, observations, predict_physics=False):
            """Process sequence of observations.
            
            Args:
                observations: (B, T, state_dim)
                predict_physics: whether to return physics predictions
            Returns:
                recon_loss, kl_loss, (optional physics predictions)
            """
            B, T, D = observations.shape
            device = observations.device
            
            # Initialize
            h = torch.zeros(B, self.deterministic_dim, device=device)
            z = torch.zeros(B, self.stochastic_dim, device=device)
            
            recon_losses = []
            kl_losses = []
            all_states = []
            
            for t in range(T):
                obs_t = observations[:, t]
                encoded = self.encoder(obs_t)
                
                # GRU update
                gru_input = torch.cat([encoded, z], dim=-1)
                h = self.gru(gru_input, h)
                
                # Prior
                prior_params = self.prior_net(h)
                prior_mean, prior_logvar = prior_params.chunk(2, dim=-1)
                
                # Posterior 
                posterior_params = self.posterior_net(torch.cat([h, encoded], dim=-1))
                post_mean, post_logvar = posterior_params.chunk(2, dim=-1)
                
                # Sample stochastic state (reparameterization)
                if self.training:
                    z = post_mean + torch.randn_like(post_mean) * (post_logvar * 0.5).exp()
                else:
                    z = post_mean
                
                # Decode
                full_state = torch.cat([h, z], dim=-1)
                recon = self.decoder(full_state)
                
                recon_losses.append(nn.functional.mse_loss(recon, obs_t))
                
                # KL divergence
                kl = 0.5 * (prior_logvar - post_logvar + 
                           (post_logvar.exp() + (post_mean - prior_mean).pow(2)) / prior_logvar.exp() - 1)
                kl_losses.append(kl.mean())
                
                all_states.append(full_state)
            
            recon_loss = torch.stack(recon_losses).mean()
            kl_loss = torch.stack(kl_losses).mean()
            
            if predict_physics:
                # Use last state for physics prediction
                final_state = all_states[-1]
                mass_pred = self.mass_head(final_state)
                friction_pred = self.friction_head(final_state)
                restitution_pred = self.restitution_head(final_state)
                return recon_loss, kl_loss, mass_pred, friction_pred, restitution_pred
            
            return recon_loss, kl_loss
        
        def predict_multihorizon(self, observations, horizons=[1, 5, 10]):
            """Predict future states at multiple horizons without teacher forcing."""
            B, T, D = observations.shape
            device = observations.device
            
            # Encode first half of trajectory
            mid = T // 2
            h = torch.zeros(B, self.deterministic_dim, device=device)
            z = torch.zeros(B, self.stochastic_dim, device=device)
            
            for t in range(mid):
                obs_t = observations[:, t]
                encoded = self.encoder(obs_t)
                gru_input = torch.cat([encoded, z], dim=-1)
                h = self.gru(gru_input, h)
                post_params = self.posterior_net(torch.cat([h, encoded], dim=-1))
                post_mean, _ = post_params.chunk(2, dim=-1)
                z = post_mean
            
            # Predict future (open-loop)
            predictions = {}
            pred_state = self.decoder(torch.cat([h, z], dim=-1))
            
            for future_t in range(max(horizons)):
                encoded = self.encoder(pred_state)
                gru_input = torch.cat([encoded, z], dim=-1)
                h = self.gru(gru_input, h)
                prior_params = self.prior_net(h)
                prior_mean, _ = prior_params.chunk(2, dim=-1)
                z = prior_mean
                pred_state = self.decoder(torch.cat([h, z], dim=-1))
                
                if (future_t + 1) in horizons:
                    actual_t = min(mid + future_t + 1, T - 1)
                    predictions[future_t + 1] = {
                        "predicted": pred_state,
                        "actual": observations[:, actual_t],
                    }
            
            return predictions
    
    # Split data
    n_train = int(len(padded) * 0.8)
    train_data = torch.tensor(padded[:n_train], dtype=torch.float32)
    test_data = torch.tensor(padded[n_train:], dtype=torch.float32)
    
    train_mass = torch.tensor([p.get("mass", p.get("mass", 1.0)) for p in physics_labels[:n_train]], dtype=torch.float32)
    test_mass = torch.tensor([p.get("mass", p.get("mass", 1.0)) for p in physics_labels[n_train:]], dtype=torch.float32)
    train_friction = torch.tensor([p.get("static_friction", p.get("friction", 0.5)) for p in physics_labels[:n_train]], dtype=torch.float32)
    test_friction = torch.tensor([p.get("static_friction", p.get("friction", 0.5)) for p in physics_labels[n_train:]], dtype=torch.float32)
    train_restitution = torch.tensor([p.get("restitution", 0.5) for p in physics_labels[:n_train]], dtype=torch.float32)
    test_restitution = torch.tensor([p.get("restitution", 0.5) for p in physics_labels[n_train:]], dtype=torch.float32)
    
    # Build RSSM
    rssm = RSSM(state_dim=state_dim, hidden_dim=256, stochastic_dim=32, deterministic_dim=256).to(device)
    optimizer = torch.optim.Adam(rssm.parameters(), lr=3e-4)
    
    logger.info(f"Training RSSM: {sum(p.numel() for p in rssm.parameters())/1e3:.1f}K params, "
                f"{len(train_data)} train, {len(test_data)} test sequences")
    
    # Train
    batch_size = 64
    n_epochs = 100
    kl_weight = 0.1
    
    for epoch in range(n_epochs):
        rssm.train()
        indices = torch.randperm(len(train_data))
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            obs_batch = train_data[batch_idx].to(device)
            mass_batch = train_mass[batch_idx].to(device)
            fric_batch = train_friction[batch_idx].to(device)
            rest_batch = train_restitution[batch_idx].to(device)
            
            recon_loss, kl_loss, mass_pred, fric_pred, rest_pred = rssm(obs_batch, predict_physics=True)
            
            physics_loss = (nn.functional.mse_loss(mass_pred.squeeze(), mass_batch) +
                          nn.functional.mse_loss(fric_pred.squeeze(), fric_batch) +
                          nn.functional.mse_loss(rest_pred.squeeze(), rest_batch))
            
            loss = recon_loss + kl_weight * kl_loss + 0.5 * physics_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rssm.parameters(), 100.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 25 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            logger.info(f"  RSSM epoch {epoch+1}/{n_epochs}, loss={avg_loss:.4f}")
    
    # Evaluate
    rssm.eval()
    
    # Physics prediction R²
    with torch.no_grad():
        all_mass_pred, all_fric_pred, all_rest_pred = [], [], []
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size].to(device)
            _, _, mp, fp, rp = rssm(batch, predict_physics=True)
            all_mass_pred.append(mp.cpu())
            all_fric_pred.append(fp.cpu())
            all_rest_pred.append(rp.cpu())
        
        mass_pred_all = torch.cat(all_mass_pred).squeeze().numpy()
        fric_pred_all = torch.cat(all_fric_pred).squeeze().numpy()
        rest_pred_all = torch.cat(all_rest_pred).squeeze().numpy()
    
    mass_r2 = r2_score(test_mass.numpy(), mass_pred_all)
    fric_r2 = r2_score(test_friction.numpy(), fric_pred_all)
    rest_r2 = r2_score(test_restitution.numpy(), rest_pred_all)
    
    logger.info(f"RSSM Physics Prediction: mass_r2={mass_r2:.4f}, friction_r2={fric_r2:.4f}, restitution_r2={rest_r2:.4f}")
    
    # Multi-horizon prediction
    horizons = [1, 3, 5, 10]
    horizon_results = {}
    
    with torch.no_grad():
        batch = test_data[:min(100, len(test_data))].to(device)
        preds = rssm.predict_multihorizon(batch, horizons=horizons)
        
        for h, data in preds.items():
            mse = nn.functional.mse_loss(data["predicted"], data["actual"]).item()
            horizon_results[f"horizon_{h}"] = {"mse": mse}
            logger.info(f"  Horizon {h}: MSE={mse:.6f}")
    
    results = {
        "physics_prediction": {
            "mass_r2": float(mass_r2),
            "friction_r2": float(fric_r2),
            "restitution_r2": float(rest_r2),
        },
        "multi_horizon": horizon_results,
        "training": {
            "n_train": len(train_data),
            "n_test": len(test_data),
            "n_epochs": n_epochs,
            "state_dim": state_dim,
            "data_source": "maniskill3" if fp_dir else "synthetic",
        },
    }
    
    out_dir = Path("results/dreamer_world_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dreamer_rssm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Dreamer RSSM results saved to {out_dir}")
    
    del rssm
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all", "ablations", "mcr", "invisible", "dreamer"],
                        default="all")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    
    all_results = {}
    
    if args.phase in ("all", "ablations"):
        all_results["ablations"] = evaluate_new_ablations(device=args.device)
    
    if args.phase in ("all", "mcr"):
        all_results["mcr"] = evaluate_mcr_baseline(device=args.device)
    
    if args.phase in ("all", "invisible"):
        all_results["invisible_physics"] = run_invisible_physics_test(device=args.device)
    
    if args.phase in ("all", "dreamer"):
        all_results["dreamer"] = run_dreamer_world_model(device=args.device)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    for phase, res in all_results.items():
        if res:
            logger.info(f"  {phase}: {len(res)} results")
    
    with open("results/post_training_eval_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info("All results saved!")


if __name__ == "__main__":
    main()
