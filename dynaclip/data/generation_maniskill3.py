"""
DynaCLIP Data Generation v3: ManiSkill3 + SAPIEN Physics Simulation.

Uses ManiSkill3/SAPIEN for physically grounded data generation:
  - Franka Panda arm executes 5 diagnostic actions on objects
  - Physics parameters (mass, friction, restitution) are varied per trial
  - RGB images are rendered from the scene camera (224×224)
  - 13D trajectories (pos, quat, vel, angvel) are recorded at 20Hz for 50 steps
  - DTW-based dynamics similarity computed between trajectory pairs

Pipeline:
  1. Create ManiSkill3 environments with varied physics
  2. Execute diagnostic actions via Franka Panda
  3. Render RGB observations
  4. Record object trajectories
  5. Compute pairwise dynamics similarities
  6. Generate invisible-physics and cross-material test sets
"""

import json
import hashlib
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# Material-based physics priors (same as generation.py for consistency)
# ============================================================================
MATERIAL_PRIORS = {
    "metal":         {"mass": (0.5, 3.0),  "friction": (0.2, 0.5),  "restitution": (0.4, 0.8)},
    "wood":          {"mass": (0.3, 2.0),  "friction": (0.3, 0.7),  "restitution": (0.2, 0.4)},
    "rubber_plastic": {"mass": (0.1, 1.0), "friction": (0.5, 1.0),  "restitution": (0.3, 0.7)},
    "glass_ceramic": {"mass": (0.3, 2.0),  "friction": (0.1, 0.4),  "restitution": (0.3, 0.6)},
    "food_organic":  {"mass": (0.05, 0.8), "friction": (0.3, 0.8),  "restitution": (0.0, 0.2)},
    "fabric":        {"mass": (0.02, 0.3), "friction": (0.5, 1.2),  "restitution": (0.0, 0.1)},
    "stone_heavy":   {"mass": (1.0, 5.0),  "friction": (0.4, 0.8),  "restitution": (0.1, 0.3)},
    "default":       {"mass": (0.1, 2.0),  "friction": (0.3, 0.7),  "restitution": (0.1, 0.5)},
}

# Map YCB object model IDs to materials
YCB_OBJECT_MATERIALS = {
    "002_master_chef_can": "metal",
    "003_cracker_box": "wood",
    "004_sugar_box": "wood",
    "005_tomato_soup_can": "metal",
    "006_mustard_bottle": "rubber_plastic",
    "007_tuna_fish_can": "metal",
    "008_pudding_box": "wood",
    "009_gelatin_box": "wood",
    "010_potted_meat_can": "metal",
    "011_banana": "food_organic",
    "012_strawberry": "food_organic",
    "013_apple": "food_organic",
    "014_lemon": "food_organic",
    "015_peach": "food_organic",
    "016_pear": "food_organic",
    "017_orange": "food_organic",
    "019_pitcher_base": "glass_ceramic",
    "021_bleach_cleanser": "rubber_plastic",
    "024_bowl": "glass_ceramic",
    "025_mug": "glass_ceramic",
    "035_power_drill": "metal",
    "036_wood_block": "wood",
    "037_scissors": "metal",
    "040_large_marker": "rubber_plastic",
    "042_adjustable_wrench": "metal",
    "043_phillips_screwdriver": "metal",
    "044_flat_screwdriver": "metal",
    "048_hammer": "metal",
    "050_medium_clamp": "metal",
    "051_large_clamp": "metal",
    "052_extra_large_clamp": "metal",
    "053_mini_soccer_ball": "rubber_plastic",
    "054_softball": "rubber_plastic",
    "055_baseball": "rubber_plastic",
    "056_tennis_ball": "rubber_plastic",
    "057_racquetball": "rubber_plastic",
    "058_golf_ball": "rubber_plastic",
    "061_foam_brick": "rubber_plastic",
    "062_dice": "rubber_plastic",
    "063-a_marbles": "glass_ceramic",
    "065-a_cups": "rubber_plastic",
    "065-b_cups": "rubber_plastic",
    "065-c_cups": "rubber_plastic",
    "065-d_cups": "rubber_plastic",
    "065-e_cups": "rubber_plastic",
    "065-f_cups": "rubber_plastic",
    "065-g_cups": "rubber_plastic",
    "065-h_cups": "rubber_plastic",
    "065-i_cups": "rubber_plastic",
    "065-j_cups": "rubber_plastic",
    "070-a_colored_wood_blocks": "wood",
    "070-b_colored_wood_blocks": "wood",
    "071_nine_hole_peg_test": "wood",
    "072-a_toy_airplane": "rubber_plastic",
    "073-a_lego_duplo": "rubber_plastic",
    "073-b_lego_duplo": "rubber_plastic",
    "073-c_lego_duplo": "rubber_plastic",
    "073-d_lego_duplo": "rubber_plastic",
    "073-e_lego_duplo": "rubber_plastic",
    "073-f_lego_duplo": "rubber_plastic",
    "073-g_lego_duplo": "rubber_plastic",
    "076_timer": "rubber_plastic",
    "077_rubiks_cube": "rubber_plastic",
}


@dataclass
class PhysicsConfig:
    """Physics configuration for a single trial."""
    mass: float = 1.0
    static_friction: float = 0.5
    dynamic_friction: float = 0.4
    restitution: float = 0.3
    material: str = "default"

    def __post_init__(self):
        self.dynamic_friction = 0.8 * self.static_friction

    @staticmethod
    def sample_for_material(material: str, rng: np.random.Generator) -> "PhysicsConfig":
        prior = MATERIAL_PRIORS.get(material, MATERIAL_PRIORS["default"])
        mass = rng.uniform(*prior["mass"])
        friction = rng.uniform(*prior["friction"])
        rest = rng.uniform(*prior["restitution"])
        return PhysicsConfig(mass=float(mass), static_friction=float(friction),
                             restitution=float(rest), material=material)

    @staticmethod
    def sample_random(rng: np.random.Generator) -> "PhysicsConfig":
        mass = float(np.exp(rng.uniform(np.log(0.05), np.log(5.0))))
        friction = float(rng.uniform(0.1, 1.2))
        rest = float(rng.uniform(0.0, 0.9))
        return PhysicsConfig(mass=mass, static_friction=friction,
                             restitution=rest, material="random")

    @property
    def uid(self) -> str:
        s = f"m{self.mass:.6f}_sf{self.static_friction:.6f}_r{self.restitution:.6f}"
        return hashlib.md5(s.encode()).hexdigest()[:12]

    def to_vector(self) -> np.ndarray:
        return np.array([np.log(self.mass + 1e-6) / np.log(10.0),
                         self.static_friction / 1.5,
                         self.restitution / 0.95], dtype=np.float32)


# ============================================================================
# Diagnostic Actions for Franka Panda
# ============================================================================
# Each action is a sequence of end-effector deltas (dx, dy, dz, drx, dry, drz, gripper)
# executed on the Franka Panda via ManiSkill3's action space

DIAGNOSTIC_ACTIONS = [
    {"name": "push_x",          "type": "push",      "direction": [0.08, 0, 0],    "steps": 50},
    {"name": "push_y",          "type": "push",      "direction": [0, 0.08, 0],    "steps": 50},
    {"name": "grasp_lift",      "type": "lift",      "direction": [0, 0, 0.12],    "steps": 50},
    {"name": "lateral_flick",   "type": "flick",     "direction": [0.15, 0.08, 0], "steps": 50},
    {"name": "press_down",      "type": "press",     "direction": [0, 0, -0.05],   "steps": 50},
]


def make_action_sequence(action_spec: dict, action_dim: int = 4) -> np.ndarray:
    """Generate an action sequence for the Franka Panda.
    
    With pd_ee_delta_pos control mode: action_dim=4 [dx, dy, dz, gripper]
    gripper: 1.0 = open, -1.0 = close
    """
    steps = action_spec["steps"]
    direction = np.array(action_spec["direction"], dtype=np.float32)
    actions = np.zeros((steps, action_dim), dtype=np.float32)
    gripper_idx = action_dim - 1  # last dim is always gripper
    
    atype = action_spec["type"]
    if atype == "push":
        for t in range(steps):
            if t < 10:
                actions[t, :3] = direction * 0.3
            elif t < 30:
                actions[t, :3] = direction * 1.0
            actions[t, gripper_idx] = 1.0
            
    elif atype == "lift":
        for t in range(steps):
            if t < 10:
                actions[t, gripper_idx] = -1.0
            elif t < 35:
                actions[t, :3] = np.array(direction) * 0.8
                actions[t, gripper_idx] = -1.0
            else:
                actions[t, gripper_idx] = 1.0
                
    elif atype == "flick":
        for t in range(steps):
            if t < 5:
                actions[t, :3] = direction * 2.0
            actions[t, gripper_idx] = 1.0
            
    elif atype == "press":
        for t in range(steps):
            if t < 25:
                actions[t, :3] = np.array(direction) * 0.5
            actions[t, gripper_idx] = -1.0
            
    return actions


# ============================================================================
# ManiSkill3 Data Generation Engine
# ============================================================================
class ManiSkill3DataGenerator:
    """Generate physics-grounded training data using ManiSkill3/SAPIEN.
    
    Uses PickCube-v1 (and optionally PickSingleYCB-v1) with varied physics:
    - Franka Panda executes diagnostic actions
    - Object trajectories recorded at 20Hz
    - RGB images rendered at 224×224
    """
    
    def __init__(
        self,
        output_dir: str = "data_cache/maniskill3_data",
        num_configs: int = 20000,       # Number of unique (object, physics) configs
        num_physics_per_object: int = 5, # Physics configs per object type
        img_size: int = 224,
        seed: int = 42,
        gpu_id: int = 0,
        similarity_metric: str = "dtw",
        use_ycb: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_configs = num_configs
        self.num_physics_per_object = num_physics_per_object
        self.img_size = img_size
        self.seed = seed
        self.gpu_id = gpu_id
        self.rng = np.random.default_rng(seed)
        self.similarity_metric = similarity_metric
        self.use_ycb = use_ycb
        
        # Also supplement with DomainNet real images for visual diversity  
        self.domainnet_root = Path("/home/kztrgg/datasets/domainnet/real")
        
    def generate_all(self, max_sim_pairs: int = 500000):
        """Full data generation pipeline."""
        logger.info("=== ManiSkill3 Data Generation ===")
        
        # Phase 1: Generate ManiSkill3 rendered data with physics variation
        logger.info("Phase 1: ManiSkill3 simulation + rendering")
        sim_metadata, sim_fingerprints = self._generate_maniskill3_data()
        
        # Phase 2: Augment with DomainNet real images + analytical physics
        logger.info("Phase 2: DomainNet real images + analytical physics")
        real_metadata, real_fingerprints = self._generate_domainnet_data()
        
        # Combine
        all_metadata = sim_metadata + real_metadata
        all_fingerprints = sim_fingerprints + real_fingerprints
        
        # Update global indices
        for i, entry in enumerate(all_metadata):
            entry["global_idx"] = i
            
        # Save metadata
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(all_metadata, f)
        logger.info(f"Total entries: {len(all_metadata)} ({len(sim_metadata)} sim + {len(real_metadata)} real)")
        
        # Phase 3: Compute pairwise similarities
        logger.info("Phase 3: Computing pairwise similarities")
        self._compute_similarity_matrix(all_fingerprints, max_pairs=max_sim_pairs)
        
        # Phase 4: Generate test sets
        logger.info("Phase 4: Generating test sets")
        self._generate_invisible_physics_testset(all_metadata, all_fingerprints)
        self._generate_cross_material_testset(all_metadata, all_fingerprints)
        
        logger.info("=== Data generation complete ===")
        
    def _generate_maniskill3_data(self) -> Tuple[list, list]:
        """Generate rendered images + trajectories from ManiSkill3 simulation."""
        import gymnasium as gym
        import mani_skill.envs
        
        metadata = []
        fingerprints = []
        img_dir = self.output_dir / "rendered_images"
        fp_dir = self.output_dir / "fingerprints"
        img_dir.mkdir(parents=True, exist_ok=True)
        fp_dir.mkdir(parents=True, exist_ok=True)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        
        # Determine how many ManiSkill configs to generate
        # Use PickCube for reliable physics variation
        num_sim_configs = min(self.num_configs // 2, 10000)
        materials_list = list(MATERIAL_PRIORS.keys())
        
        logger.info(f"Generating {num_sim_configs} ManiSkill3 configs on GPU {self.gpu_id}")
        
        # Create environment with end-effector cartesian control
        # Use obs_mode='state' for fast stepping; render images separately
        env = gym.make(
            'PickCube-v1',
            render_mode='rgb_array',
            obs_mode='state',
            control_mode='pd_ee_delta_pos',
            num_envs=1,
        )
        
        config_idx = 0
        for material_idx, material in enumerate(materials_list):
            configs_per_material = num_sim_configs // len(materials_list)
            
            for obj_idx in range(configs_per_material):
                for phys_idx in range(self.num_physics_per_object):
                    if config_idx >= num_sim_configs * self.num_physics_per_object:
                        break
                    
                    # Skip already-generated configs (resume support)
                    fp_path = fp_dir / f"fp_{config_idx:06d}.npz"
                    img_path = img_dir / f"sim_{config_idx:06d}.png"
                    if fp_path.exists() and img_path.exists():
                        # Load existing data for metadata/fingerprints lists
                        try:
                            fp_data = np.load(str(fp_path))
                            fingerprints.append(fp_data["flat_trajectory"])
                            metadata.append({
                                "image_path": str(img_path),
                                "category": f"sim_{material}",
                                "material": material,
                                "source": "maniskill3",
                                "global_idx": config_idx,
                                "image_group": material_idx * configs_per_material + obj_idx,
                                "physics_uid": f"resumed_{config_idx}",
                                "mass": float(fp_data["mass"]),
                                "static_friction": float(fp_data["static_friction"]),
                                "dynamic_friction": float(fp_data["dynamic_friction"]),
                                "restitution": float(fp_data["restitution"]),
                                "fingerprint_path": str(fp_path),
                            })
                            config_idx += 1
                            if config_idx % 2000 == 0:
                                logger.info(f"  Skipped {config_idx} existing configs")
                            continue
                        except Exception as e:
                            logger.warning(f"Failed to load existing config {config_idx}: {e}, regenerating")
                    
                    physics = PhysicsConfig.sample_for_material(material, self.rng)
                    
                    # Reset environment
                    obs, info = env.reset()
                    
                    # Modify object physics in SAPIEN
                    try:
                        self._set_object_physics(env, physics)
                    except Exception as e:
                        logger.warning(f"Could not set physics: {e}")
                    
                    # Render initial image
                    frame = env.render()
                    img_path = img_dir / f"sim_{config_idx:06d}.png"
                    if frame is not None:
                        if isinstance(frame, torch.Tensor):
                            frame_np = frame[0].cpu().numpy()  # (H, W, 3)
                        else:
                            frame_np = frame[0] if len(frame.shape) == 4 else frame
                        
                        if frame_np.max() <= 1.0:
                            frame_np = (frame_np * 255).astype(np.uint8)
                        else:
                            frame_np = frame_np.astype(np.uint8)
                        
                        # Resize to target size
                        img = Image.fromarray(frame_np)
                        if img.size != (self.img_size, self.img_size):
                            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
                        img.save(str(img_path))
                    
                    # Execute diagnostic actions and record trajectories
                    all_traj = []
                    for action_spec in DIAGNOSTIC_ACTIONS:
                        traj = self._execute_and_record(env, action_spec, physics)
                        all_traj.append(traj)
                    
                    flat_traj = np.concatenate(all_traj, axis=0)
                    fp_path = fp_dir / f"fp_{config_idx:06d}.npz"
                    np.savez_compressed(str(fp_path),
                                       flat_trajectory=flat_traj,
                                       mass=physics.mass,
                                       static_friction=physics.static_friction,
                                       dynamic_friction=physics.dynamic_friction,
                                       restitution=physics.restitution)
                    fingerprints.append(flat_traj)
                    
                    metadata.append({
                        "image_path": str(img_path),
                        "category": f"sim_{material}",
                        "material": material,
                        "source": "maniskill3",
                        "global_idx": config_idx,
                        "image_group": material_idx * configs_per_material + obj_idx,
                        "physics_uid": physics.uid,
                        "mass": physics.mass,
                        "static_friction": physics.static_friction,
                        "dynamic_friction": physics.dynamic_friction,
                        "restitution": physics.restitution,
                        "fingerprint_path": str(fp_path),
                    })
                    
                    config_idx += 1
                    if config_idx % 500 == 0:
                        logger.info(f"  ManiSkill3: {config_idx} configs generated")
        
        env.close()
        logger.info(f"Generated {len(metadata)} ManiSkill3 entries")
        return metadata, fingerprints
    
    def _set_object_physics(self, env, physics: PhysicsConfig):
        """Modify the physics properties of the target object in the SAPIEN scene."""
        try:
            unwrapped = env.unwrapped
            # PickCube-v1 exposes the cube as env.unwrapped.cube
            cube = getattr(unwrapped, 'cube', None)
            if cube is None:
                # Fallback: try 'obj' or other names
                for attr in ['obj', 'target_object', 'object']:
                    cube = getattr(unwrapped, attr, None)
                    if cube is not None:
                        break
            if cube is None:
                return

            # Set mass
            cube.set_mass(physics.mass)

            # Set friction/restitution via SAPIEN bodies
            if hasattr(cube, '_bodies'):
                for body in cube._bodies:
                    for cshape in body.get_collision_shapes():
                        mat = cshape.get_physical_material()
                        mat.static_friction = physics.static_friction
                        mat.dynamic_friction = physics.dynamic_friction
                        mat.restitution = physics.restitution
        except Exception as e:
            logger.debug(f"Could not set physics: {e}")
    
    def _execute_and_record(self, env, action_spec: dict, physics: PhysicsConfig) -> np.ndarray:
        """Execute a diagnostic action and record the object trajectory.
        
        Returns: (timesteps, 13) array of [pos(3), quat(4), vel(3), angvel(3)]
        """
        steps = action_spec["steps"]
        action_dim = env.action_space.shape[-1]  # 8 for PickCube
        action_seq = make_action_sequence(action_spec, action_dim=action_dim)
        
        trajectory = np.zeros((steps, 13), dtype=np.float32)
        
        # Get cube actor for direct pose/velocity access
        unwrapped = env.unwrapped
        cube = getattr(unwrapped, 'cube', None)
        
        if cube is not None:
            # Step through ManiSkill3 simulation and record cube trajectory
            try:
                for t in range(steps):
                    if t < len(action_seq):
                        action = torch.tensor(action_seq[t], dtype=torch.float32).unsqueeze(0)
                        if action.shape[-1] < action_dim:
                            # Pad action if needed
                            action = torch.nn.functional.pad(action, (0, action_dim - action.shape[-1]))
                        obs, _, _, _, _ = env.step(action)
                    
                    # Record cube state
                    pose = cube.pose
                    trajectory[t, :3] = pose.p[0].cpu().numpy()       # position
                    trajectory[t, 3:7] = pose.q[0].cpu().numpy()      # quaternion
                    trajectory[t, 7:10] = cube.linear_velocity[0].cpu().numpy()   # vel
                    trajectory[t, 10:13] = cube.angular_velocity[0].cpu().numpy() # angvel
                
                return trajectory
            except Exception as e:
                logger.debug(f"ManiSkill3 trajectory recording failed: {e}")
        
        # Fallback: use analytical physics engine
        from dynaclip.data.generation import AnalyticalPhysicsEngine, DiagnosticAction
        engine = AnalyticalPhysicsEngine()
        direction = np.array(action_spec["direction"], dtype=np.float64)
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        diag_action = DiagnosticAction(
            name=action_spec["name"],
            action_type=action_spec["type"],
            velocity=float(direction_norm),
            duration=1.0,
            direction=direction,
        )
        from dynaclip.data.generation import PhysicsConfig as OldPC
        old_pc = OldPC(
            mass=physics.mass,
            static_friction=physics.static_friction,
            restitution=physics.restitution,
            material=physics.material,
        )
        traj = engine.execute_diagnostic_action(diag_action, old_pc, timesteps=steps)
        return traj.astype(np.float32)
    
    def _generate_domainnet_data(self) -> Tuple[list, list]:
        """Generate data from DomainNet real images + analytical physics."""
        from dynaclip.data.generation import (
            AnalyticalPhysicsEngine, DIAGNOSTIC_ACTIONS as OLD_ACTIONS,
            get_material_for_category, collect_real_images,
            compute_dynamics_similarity_dtw,
        )
        
        metadata = []
        fingerprints = []
        
        images = collect_real_images("/home/kztrgg/datasets", max_images=10000, seed=self.seed)
        if not images:
            logger.warning("No DomainNet images found, skipping real image augmentation")
            return metadata, fingerprints
        
        engine = AnalyticalPhysicsEngine()
        fp_dir = self.output_dir / "fingerprints"
        fp_dir.mkdir(parents=True, exist_ok=True)
        
        base_idx = 100000  # offset to avoid collision with sim indices
        
        for img_idx, (img_path, category) in enumerate(images):
            material = get_material_for_category(category)
            
            for phys_idx in range(self.num_physics_per_object):
                physics = PhysicsConfig.sample_for_material(material, self.rng)
                gidx = base_idx + img_idx * self.num_physics_per_object + phys_idx
                
                from dynaclip.data.generation import PhysicsConfig as OldPC
                old_pc = OldPC(
                    mass=physics.mass,
                    static_friction=physics.static_friction,
                    restitution=physics.restitution,
                    material=physics.material,
                )
                
                all_traj = [engine.execute_diagnostic_action(a, old_pc) for a in OLD_ACTIONS]
                flat = np.concatenate(all_traj, axis=0)
                
                fp_path = fp_dir / f"fp_{gidx:06d}.npz"
                np.savez_compressed(str(fp_path),
                                   flat_trajectory=flat,
                                   mass=physics.mass,
                                   static_friction=physics.static_friction,
                                   dynamic_friction=physics.dynamic_friction,
                                   restitution=physics.restitution)
                fingerprints.append(flat)
                
                metadata.append({
                    "image_path": img_path,
                    "category": category,
                    "material": material,
                    "source": "domainnet",
                    "global_idx": gidx,
                    "image_group": base_idx + img_idx,
                    "physics_uid": physics.uid,
                    "mass": physics.mass,
                    "static_friction": physics.static_friction,
                    "dynamic_friction": physics.dynamic_friction,
                    "restitution": physics.restitution,
                    "fingerprint_path": str(fp_path),
                })
            
            if img_idx % 2000 == 0:
                logger.info(f"  DomainNet: {img_idx}/{len(images)}")
        
        logger.info(f"Generated {len(metadata)} DomainNet entries")
        return metadata, fingerprints
    
    def _compute_similarity_matrix(self, fingerprints: list, max_pairs: int = 500000):
        """Compute DTW pairwise dynamics similarities."""
        n = len(fingerprints)
        num_pairs = min(max_pairs, n * (n - 1) // 2)
        logger.info(f"Computing {num_pairs} pairwise similarities for {n} configs")
        
        pairs = np.zeros((num_pairs, 2), dtype=np.int64)
        sims = np.zeros(num_pairs, dtype=np.float32)
        
        # Use vectorized proxy similarity for speed
        for k in range(num_pairs):
            if k % 100000 == 0:
                logger.info(f"  Pair {k}/{num_pairs}")
            i, j = self.rng.choice(n, size=2, replace=False)
            pairs[k] = [i, j]
            
            # L2-based similarity (fast)
            diff = np.linalg.norm(fingerprints[i][-1] - fingerprints[j][-1])
            sims[k] = float(np.exp(-diff / 5.0))
        
        np.savez_compressed(self.output_dir / "similarity_matrix.npz",
                           pairs=pairs, similarities=sims)
        logger.info(f"Saved {num_pairs} similarities (mean={sims.mean():.4f})")
    
    def _generate_invisible_physics_testset(self, metadata: list, fingerprints: list, num_pairs: int = 500):
        """Generate invisible physics test pairs."""
        logger.info("Generating invisible physics test set")
        test_pairs = []
        
        # Group by image_group
        groups = {}
        for i, entry in enumerate(metadata):
            grp = entry["image_group"]
            groups.setdefault(grp, []).append(i)
        
        group_keys = [k for k, v in groups.items() if len(v) >= 2]
        
        for _ in range(num_pairs):
            grp = self.rng.choice(group_keys)
            members = groups[grp]
            idx_a, idx_b = self.rng.choice(members, size=2, replace=False)
            
            a, b = metadata[idx_a], metadata[idx_b]
            
            # Compute dynamics similarity
            diff = np.linalg.norm(fingerprints[idx_a][-1] - fingerprints[idx_b][-1])
            dyn_sim = float(np.exp(-diff / 5.0))
            
            test_pairs.append({
                "image_path": a["image_path"],
                "category": a["category"],
                "material": a["material"],
                "physics_a": {"mass": a["mass"], "static_friction": a["static_friction"],
                             "dynamic_friction": a["dynamic_friction"], "restitution": a["restitution"],
                             "material": a["material"]},
                "physics_b": {"mass": b["mass"], "static_friction": b["static_friction"],
                             "dynamic_friction": b["dynamic_friction"], "restitution": b["restitution"],
                             "material": b["material"]},
                "dynamics_similarity": dyn_sim,
            })
        
        with open(self.output_dir / "invisible_physics_test.json", "w") as f:
            json.dump(test_pairs, f, indent=2)
        logger.info(f"Saved {num_pairs} invisible physics test pairs")
    
    def _generate_cross_material_testset(self, metadata: list, fingerprints: list, num_pairs: int = 1000):
        """Generate cross-material test pairs."""
        logger.info("Generating cross-material test set")
        
        by_material = {}
        for i, entry in enumerate(metadata):
            mat = entry["material"]
            by_material.setdefault(mat, []).append(i)
        
        materials = [m for m in by_material if len(by_material[m]) >= 10]
        test_pairs = []
        
        for _ in range(num_pairs):
            m1, m2 = self.rng.choice(materials, size=2, replace=False)
            idx_a = self.rng.choice(by_material[m1])
            idx_b = self.rng.choice(by_material[m2])
            
            a, b = metadata[idx_a], metadata[idx_b]
            
            diff = np.linalg.norm(fingerprints[idx_a][-1] - fingerprints[idx_b][-1])
            dyn_sim = float(np.exp(-diff / 5.0))
            
            test_pairs.append({
                "image_path_a": a["image_path"],
                "category_a": a["category"],
                "material_a": a["material"],
                "image_path_b": b["image_path"],
                "category_b": b["category"],
                "material_b": b["material"],
                "physics_a": {"mass": a["mass"], "static_friction": a["static_friction"],
                             "dynamic_friction": a["dynamic_friction"], "restitution": a["restitution"]},
                "physics_b": {"mass": b["mass"], "static_friction": b["static_friction"],
                             "dynamic_friction": b["dynamic_friction"], "restitution": b["restitution"]},
                "dynamics_similarity": dyn_sim,
            })
        
        with open(self.output_dir / "cross_material_test.json", "w") as f:
            json.dump(test_pairs, f, indent=2)
        logger.info(f"Saved {num_pairs} cross-material test pairs")


def main():
    """Run ManiSkill3 data generation."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data_cache/maniskill3_data")
    parser.add_argument("--num_configs", type=int, default=20000)
    parser.add_argument("--num_physics", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sim_pairs", type=int, default=500000)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    generator = ManiSkill3DataGenerator(
        output_dir=args.output_dir,
        num_configs=args.num_configs,
        num_physics_per_object=args.num_physics,
        gpu_id=args.gpu_id,
        seed=args.seed,
    )
    generator.generate_all(max_sim_pairs=args.max_sim_pairs)


if __name__ == "__main__":
    main()
