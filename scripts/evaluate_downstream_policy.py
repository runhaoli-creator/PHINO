#!/usr/bin/env python
"""
DynaCLIP Downstream Policy Evaluation on ManiSkill3.

Experiment: Train a simple BC (Behavior Cloning) policy head on demonstrations
collected with standard physics, then evaluate generalization when physics change.

Hypothesis: DynaCLIP features encode physics information, enabling better policy
adaptation to novel physics conditions.

Pipeline:
  1. Collect demonstrations from scripted expert on PickCube-v1 (standard physics)
  2. Extract visual features from observations using each backbone
  3. Train an MLP BC policy: features + joint state -> actions
  4. Evaluate on 5 physics variations (different mass, friction, restitution)
  5. Report success rates per backbone × physics

Backbone comparisons:
  - DynaCLIP (ours, 30K)
  - DINOv2 (baseline: frozen DINOv2 ViT-B/14)
  - CLIP (ViT-B/16)
  - R3M (ResNet-50 from R3M)
  - VIP (ResNet-50 from VIP)
  - ImageNet ViT-B/14
  - Random init ViT-B/14
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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, ".")

from dynaclip.models.dynaclip import DynaCLIPModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/downstream_policy.log"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Backbone loading helpers (from evaluate_full.py)
# ============================================================================
class BackboneWrapper(nn.Module):
    """Wraps any backbone to output a flat feature vector."""
    def __init__(self, backbone, name="backbone"):
        super().__init__()
        self.backbone = backbone
        self.name = name
    
    @torch.no_grad()
    def forward(self, x):
        out = self.backbone(x)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, 'last_hidden_state'):
            return out.last_hidden_state[:, 0]
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            return list(out.values())[0]
        if isinstance(out, torch.Tensor):
            if out.dim() == 3:
                return out[:, 0]
            return out
        if isinstance(out, (tuple, list)):
            return out[0] if out[0].dim() == 2 else out[0][:, 0]
        return out


class DynaCLIPWrapper(nn.Module):
    """Wraps trained DynaCLIP model for feature extraction."""
    def __init__(self, model, use_features=True):
        super().__init__()
        self.model = model
        self.use_features = use_features
    
    @torch.no_grad()
    def forward(self, x):
        return self.model(x, return_features=self.use_features)


def load_all_backbones(checkpoint_path: str, device: str) -> Dict[str, nn.Module]:
    """Load all backbones for comparison."""
    backbones = {}
    
    # 1. DynaCLIP
    if Path(checkpoint_path).exists():
        model = DynaCLIPModel(backbone_name="dinov2_vitb14", embed_dim=512)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        backbones["DynaCLIP"] = DynaCLIPWrapper(model, use_features=True)
        logger.info(f"Loaded DynaCLIP from {checkpoint_path}")
    
    # 2. DINOv2
    try:
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        dinov2.eval()
        backbones["DINOv2"] = BackboneWrapper(dinov2, "DINOv2")
    except Exception as e:
        logger.warning(f"DINOv2 load failed: {e}")
    
    # 3. CLIP
    try:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        clip_visual = clip_model.visual
        clip_visual.eval()
        backbones["CLIP"] = BackboneWrapper(clip_visual, "CLIP")
    except Exception as e:
        logger.warning(f"CLIP load failed: {e}")
    
    # 4. R3M-style (ResNet-50)
    try:
        import torchvision.models as tvm
        class R3MEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
                self.features = nn.Sequential(*list(resnet.children())[:-1])
            def forward(self, x):
                return self.features(x).flatten(1)
        r3m = R3MEncoder()
        r3m.eval()
        backbones["R3M"] = BackboneWrapper(r3m, "R3M")
    except Exception as e:
        logger.warning(f"R3M load failed: {e}")
    
    # 5. SigLIP
    try:
        from transformers import AutoModel
        siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        siglip_vision = siglip.vision_model
        siglip_vision.eval()
        backbones["SigLIP"] = BackboneWrapper(siglip_vision, "SigLIP")
    except Exception as e:
        logger.warning(f"SigLIP load failed: {e}")
    
    return backbones


# ============================================================================
# Physics variation configs for evaluation
# ============================================================================
PHYSICS_CONFIGS = {
    "standard": {"mass": 0.064, "friction": 0.3, "restitution": 0.0},
    "heavy": {"mass": 2.0, "friction": 0.3, "restitution": 0.0},
    "light": {"mass": 0.01, "friction": 0.3, "restitution": 0.0},
    "slippery": {"mass": 0.064, "friction": 0.05, "restitution": 0.0},
    "rough": {"mass": 0.064, "friction": 1.2, "restitution": 0.0},
    "bouncy": {"mass": 0.064, "friction": 0.3, "restitution": 0.8},
    "heavy_slippery": {"mass": 2.0, "friction": 0.05, "restitution": 0.0},
    "light_bouncy": {"mass": 0.01, "friction": 0.3, "restitution": 0.8},
}


# ============================================================================
# BC Policy (MLP)
# ============================================================================
class BCPolicy(nn.Module):
    """Simple MLP policy for Behavior Cloning.
    
    Input: visual features (D) + robot state (qpos, qvel)
    Output: end-effector delta actions (4D for pd_ee_delta_pos)
    """
    def __init__(self, feature_dim: int, state_dim: int = 42, action_dim: int = 4,
                 hidden_dims: List[int] = [512, 256, 128]):
        super().__init__()
        layers = []
        in_dim = feature_dim + state_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(in_dim, hd), nn.LayerNorm(hd), nn.GELU(), nn.Dropout(0.1)])
            in_dim = hd
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([features, state], dim=-1)
        return torch.tanh(self.net(x))  # Actions bounded in [-1, 1]


# ============================================================================
# Expert demonstration collector (scripted policy)
# ============================================================================
class ExpertCollector:
    """Collect pick-cube demonstrations using a reactive PD controller.
    
    State obs layout (42D flat tensor for obs_mode='state'):
      [0:9]   agent/qpos (9D)
      [9:18]  agent/qvel (9D) 
      [18]    extra/is_grasped (1D)
      [19:26] extra/tcp_pose (7D: xyz + quat)
      [26:29] extra/goal_pos (3D)
      [29:36] extra/obj_pose (7D: xyz + quat)
      [36:39] extra/tcp_to_obj_pos (3D)
      [39:42] extra/obj_to_goal_pos (3D)
    """
    
    def __init__(self, device: str = "cuda:0"):
        import gymnasium as gym
        import mani_skill.envs
        self.gym = gym
        self.device = device
    
    def collect_demonstrations(self, num_demos: int = 200, 
                                physics_cfg: dict = None) -> Dict[str, np.ndarray]:
        """Collect demonstrations with a reactive pick-cube policy.
        
        Returns dict with keys: observations (N, H, W, 3), states (N, 42), 
                                actions (N, 4), episode_ids (N,)
        """
        env = self.gym.make('PickCube-v1', render_mode='rgb_array', obs_mode='state',
                            control_mode='pd_ee_delta_pos', num_envs=1)
        
        all_obs = []
        all_states = []
        all_actions = []
        all_episode_ids = []
        
        successes = 0
        attempts = 0
        max_attempts = num_demos * 3
        
        while successes < num_demos and attempts < max_attempts:
            attempts += 1
            obs, _ = env.reset()
            
            # Set physics if specified
            if physics_cfg:
                cube = env.unwrapped.cube
                cube.set_mass(physics_cfg["mass"])
                for body in cube._bodies:
                    for cs in body.get_collision_shapes():
                        mat = cs.get_physical_material()
                        mat.static_friction = physics_cfg["friction"]
                        mat.dynamic_friction = physics_cfg["friction"] * 0.8
                        mat.restitution = physics_cfg["restitution"]
            
            episode_obs = []
            episode_states = []
            episode_actions = []
            success = False
            
            for t in range(200):
                # Extract state from flat obs tensor
                state_full = obs[0].cpu().numpy()  # (42,)
                tcp_pos = state_full[19:22]
                cube_pos = state_full[29:32]
                goal_pos = state_full[26:29]
                is_grasped = state_full[18]
                
                # Render image
                frame = env.render()
                if isinstance(frame, torch.Tensor):
                    frame_np = frame[0].cpu().numpy()
                else:
                    frame_np = frame[0] if len(frame.shape) == 4 else frame
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype(np.uint8)
                
                # Reactive expert action
                action = self._expert_action(tcp_pos, cube_pos, goal_pos, is_grasped)
                
                episode_obs.append(frame_np)
                episode_states.append(state_full)
                episode_actions.append(action)
                
                # Step
                action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
                obs, reward, done, truncated, info = env.step(action_t)
                
                if info['success'].item() if isinstance(info['success'], torch.Tensor) else info['success']:
                    success = True
                    break
                
                if (isinstance(done, torch.Tensor) and done.item()) or (isinstance(truncated, torch.Tensor) and truncated.item()):
                    break
            
            if success and len(episode_obs) >= 10:
                all_obs.extend(episode_obs)
                all_states.extend(episode_states)
                all_actions.extend(episode_actions)
                all_episode_ids.extend([successes] * len(episode_obs))
                successes += 1
                if successes % 50 == 0:
                    logger.info(f"  Collected {successes}/{num_demos} successful demos ({attempts} attempts)")
        
        env.close()
        
        logger.info(f"Collected {successes} successful demonstrations, {len(all_obs)} total frames")
        return {
            "observations": np.array(all_obs),      # (N, H, W, 3)
            "states": np.array(all_states),          # (N, 42)
            "actions": np.array(all_actions),        # (N, 4)
            "episode_ids": np.array(all_episode_ids),
        }
    
    def _expert_action(self, tcp_pos: np.ndarray, cube_pos: np.ndarray, 
                       goal_pos: np.ndarray, is_grasped: float) -> np.ndarray:
        """Reactive pick-cube expert policy.
        
        Strategy (state-based, no fixed timestep phases):
        1. If not grasped: approach cube from above, lower, close gripper
        2. If grasped: move to goal position
        """
        action = np.zeros(4, dtype=np.float32)
        dist_to_cube_xy = np.linalg.norm(tcp_pos[:2] - cube_pos[:2])
        
        if not is_grasped:
            if tcp_pos[2] - cube_pos[2] > 0.06 or dist_to_cube_xy > 0.02:
                # Move above cube
                target = cube_pos + np.array([0, 0, 0.05])
                diff = target - tcp_pos
                action[:3] = np.clip(diff * 10.0, -1.0, 1.0)
                action[3] = 1.0  # open gripper
            elif tcp_pos[2] - cube_pos[2] > 0.015:
                # Lower to cube 
                target = cube_pos + np.array([0, 0, 0.005])
                diff = target - tcp_pos
                action[:3] = np.clip(diff * 10.0, -1.0, 1.0)
                action[3] = 1.0  # open gripper
            else:
                # Close gripper
                action[:3] = 0.0
                action[3] = -1.0
        else:
            # Grasped — move to goal
            diff = goal_pos - tcp_pos
            action[:3] = np.clip(diff * 5.0, -1.0, 1.0)
            action[3] = -1.0  # keep closed
        
        return action


# ============================================================================
# Feature extraction
# ============================================================================
@torch.no_grad()
def extract_features_from_observations(
    observations: np.ndarray,
    backbone: nn.Module,
    device: str = "cuda:0",
    batch_size: int = 64,
) -> np.ndarray:
    """Extract visual features from observation images using a backbone.
    
    Args:
        observations: (N, H, W, 3) uint8 images
        backbone: feature extractor module
        device: CUDA device  
        batch_size: batch size for extraction
    
    Returns: (N, D) feature array
    """
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    backbone = backbone.to(device).eval()
    
    features = []
    for i in range(0, len(observations), batch_size):
        batch = observations[i:i+batch_size]
        imgs = torch.stack([transform(img) for img in batch]).to(device)
        feats = backbone(imgs)
        if isinstance(feats, tuple):
            feats = feats[0]
        if feats.dim() > 2:
            feats = feats.mean(dim=1)
        features.append(feats.cpu().numpy())
    
    backbone.cpu()
    torch.cuda.empty_cache()
    return np.concatenate(features, axis=0)


# ============================================================================
# Train BC policy
# ============================================================================
def train_bc_policy(
    features: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    feature_dim: int,
    device: str = "cuda:0",
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> BCPolicy:
    """Train a BC policy on extracted features + states -> actions."""
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    
    # Normalize features
    feat_mean = features.mean(0)
    feat_std = features.std(0) + 1e-6
    features_norm = (features - feat_mean) / feat_std
    
    # Normalize states
    state_mean = states.mean(0)
    state_std = states.std(0) + 1e-6
    states_norm = (states - state_mean) / state_std
    
    dataset = TensorDataset(
        torch.tensor(features_norm, dtype=torch.float32),
        torch.tensor(states_norm, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    policy = BCPolicy(feature_dim, state_dim, action_dim).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        total_loss = 0
        for feat_b, state_b, action_b in loader:
            feat_b, state_b, action_b = feat_b.to(device), state_b.to(device), action_b.to(device)
            pred_action = policy(feat_b, state_b)
            loss = F.mse_loss(pred_action, action_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(loader)
            logger.info(f"    BC epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")
    
    # Store normalization stats
    policy.feat_mean = torch.tensor(feat_mean, dtype=torch.float32)
    policy.feat_std = torch.tensor(feat_std, dtype=torch.float32)
    policy.state_mean = torch.tensor(state_mean, dtype=torch.float32)
    policy.state_std = torch.tensor(state_std, dtype=torch.float32)
    
    return policy


# ============================================================================
# Policy evaluation
# ============================================================================
@torch.no_grad()
def evaluate_policy(
    policy: BCPolicy,
    backbone: nn.Module,
    physics_cfg: dict,
    device: str = "cuda:0",
    num_episodes: int = 50,
    max_steps: int = 200,
) -> Dict[str, float]:
    """Evaluate BC policy on ManiSkill3 PickCube with specified physics."""
    import gymnasium as gym
    import mani_skill.envs
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    backbone = backbone.to(device).eval()
    policy = policy.to(device).eval()
    
    env = gym.make('PickCube-v1', render_mode='rgb_array', obs_mode='state',
                    control_mode='pd_ee_delta_pos', num_envs=1)
    
    successes = 0
    total_rewards = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        
        # Set physics
        cube = env.unwrapped.cube
        cube.set_mass(physics_cfg["mass"])
        for body in cube._bodies:
            for cs in body.get_collision_shapes():
                mat = cs.get_physical_material()
                mat.static_friction = physics_cfg["friction"]
                mat.dynamic_friction = physics_cfg["friction"] * 0.8
                mat.restitution = physics_cfg["restitution"]
        
        episode_reward = 0
        for t in range(max_steps):
            # Get observation image
            frame = env.render()
            if isinstance(frame, torch.Tensor):
                frame_np = frame[0].cpu().numpy()
            else:
                frame_np = frame[0] if len(frame.shape) == 4 else frame
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            
            # Get state (flat 42D tensor)
            state_full = obs[0].cpu().numpy()  # (42,)
            
            # Extract features
            img_t = transform(frame_np).unsqueeze(0).to(device)
            feat = backbone(img_t)
            if isinstance(feat, tuple):
                feat = feat[0]
            if feat.dim() > 2:
                feat = feat.mean(dim=1)
            
            # Normalize
            feat_norm = (feat.cpu() - policy.feat_mean) / policy.feat_std
            state_norm = (torch.tensor(state_full, dtype=torch.float32) - policy.state_mean) / policy.state_std
            
            # Predict action
            action = policy(feat_norm.to(device), state_norm.unsqueeze(0).to(device))
            action_np = action[0].cpu().numpy()
            
            # Step
            obs, reward, done, truncated, info = env.step(
                torch.tensor(action_np, dtype=torch.float32).unsqueeze(0)
            )
            
            r_val = reward.item() if isinstance(reward, torch.Tensor) else reward
            episode_reward += r_val
            
            success = False
            if isinstance(info, dict) and "success" in info:
                s = info["success"]
                success = s.item() if isinstance(s, torch.Tensor) else bool(s)
            
            if isinstance(done, torch.Tensor):
                done = done.item()
            if isinstance(truncated, torch.Tensor):
                truncated = truncated.item()
            
            if success or done or truncated:
                if success:
                    successes += 1
                break
        
        total_rewards.append(episode_reward)
    
    env.close()
    backbone.cpu()
    torch.cuda.empty_cache()
    
    return {
        "success_rate": successes / num_episodes,
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_demos", type=int, default=200)
    parser.add_argument("--num_eval_episodes", type=int, default=50)
    parser.add_argument("--output_dir", default="results/downstream_policy")
    parser.add_argument("--checkpoint", default="checkpoints/pretrain/dynaclip_final.pt")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    
    # ===== Phase 1: Collect demonstrations =====
    logger.info("=" * 60)
    logger.info("Phase 1: Collecting demonstrations")
    logger.info("=" * 60)
    
    demo_path = output_dir / "demonstrations.npz"
    if demo_path.exists():
        logger.info(f"Loading cached demos from {demo_path}")
        data = np.load(str(demo_path), allow_pickle=True)
        demos = {k: data[k] for k in data.files}
    else:
        collector = ExpertCollector(device=device)
        demos = collector.collect_demonstrations(
            num_demos=args.num_demos,
            physics_cfg=PHYSICS_CONFIGS["standard"],
        )
        np.savez_compressed(str(demo_path), **demos)
        logger.info(f"Saved {len(demos['observations'])} frames to {demo_path}")
    
    logger.info(f"Demo data: {demos['observations'].shape[0]} frames, "
                f"{len(np.unique(demos['episode_ids']))} episodes")
    
    # ===== Phase 2: Extract features & train BC for each backbone =====
    backbones = load_all_backbones(args.checkpoint, device)
    logger.info(f"Loaded {len(backbones)} backbones: {list(backbones.keys())}")
    
    all_results = {}
    
    for backbone_label, backbone_module in backbones.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase 2-3: {backbone_label}")
        logger.info(f"{'='*60}")
        
        # Extract features
        logger.info(f"  Extracting features with {backbone_label}...")
        try:
            features = extract_features_from_observations(
                demos["observations"],
                backbone_module,
                device=device,
            )
        except Exception as e:
            logger.warning(f"  Failed to extract features for {backbone_label}: {e}")
            import traceback; traceback.print_exc()
            continue
        
        logger.info(f"  Features: {features.shape}")
        
        # Train BC policy
        logger.info(f"  Training BC policy on {len(features)} frames...")
        policy = train_bc_policy(
            features=features,
            states=demos["states"],
            actions=demos["actions"],
            feature_dim=features.shape[1],
            device=device,
            epochs=100,
            batch_size=256,
        )
        
        # Evaluate on each physics config
        backbone_results = {}
        for phys_name, phys_cfg in PHYSICS_CONFIGS.items():
            logger.info(f"  Evaluating on {phys_name} physics...")
            try:
                result = evaluate_policy(
                    policy=policy,
                    backbone=backbone_module,
                    physics_cfg=phys_cfg,
                    device=device,
                    num_episodes=args.num_eval_episodes,
                )
                backbone_results[phys_name] = result
                logger.info(f"    {phys_name}: success={result['success_rate']:.3f}, "
                          f"reward={result['mean_reward']:.3f}")
            except Exception as e:
                logger.warning(f"    Evaluation failed for {phys_name}: {e}")
                import traceback; traceback.print_exc()
                backbone_results[phys_name] = {"success_rate": 0.0, "mean_reward": 0.0}
        
        all_results[backbone_label] = backbone_results
        
        # Cleanup
        del policy, features
        backbone_module.cpu()
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save results
    with open(output_dir / "downstream_policy_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print formatted table
    print("\n" + "=" * 120)
    print("DOWNSTREAM POLICY EVALUATION — Success Rate (%) per Physics Condition")
    print("=" * 120)
    header = f"{'Backbone':<15}"
    for phys_name in PHYSICS_CONFIGS:
        header += f" {phys_name:<15}"
    header += f" {'Mean':<10}"
    print(header)
    print("-" * 120)
    
    for backbone_label, backbone_results in all_results.items():
        row = f"{backbone_label:<15}"
        rates = []
        for phys_name in PHYSICS_CONFIGS:
            r = backbone_results.get(phys_name, {}).get("success_rate", 0.0) * 100
            rates.append(r)
            row += f" {r:<15.1f}"
        row += f" {np.mean(rates):<10.1f}"
        print(row)
    
    print("=" * 120)
    logger.info(f"\nResults saved to {output_dir / 'downstream_policy_results.json'}")


if __name__ == "__main__":
    main()
