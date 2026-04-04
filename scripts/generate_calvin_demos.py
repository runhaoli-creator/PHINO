#!/usr/bin/env python3
"""
generate_calvin_demos.py — Generate CALVIN demonstration data using scripted policies.

Creates demo episodes in the official CALVIN format (numpy files per timestep)
for use with evaluate_calvin_v3.py.

Since the CALVIN dataset server is unreachable, we generate demonstrations
using scripted controllers for the 10 most commonly evaluated tasks.

Tasks:
  1. move_slider_left
  2. move_slider_right  
  3. open_drawer
  4. close_drawer
  5. turn_on_lightbulb
  6. turn_off_lightbulb
  7. turn_on_led
  8. turn_off_led
  9. push_red_block_right
  10. lift_red_block_table

Usage:
  python generate_calvin_demos.py --output_dir data/calvin/generated_demos --n_demos 50
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

os.environ["DISPLAY"] = ""
os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.insert(0, '/home/kztrgg/calvin_env')

from omegaconf import OmegaConf

CALVIN_ENV_DATA = '/home/kztrgg/calvin_env/data'

def make_env(seed=0):
    """Create CALVIN PlayTable environment with Scene D."""
    original_dir = os.getcwd()
    os.chdir(CALVIN_ENV_DATA)

    cam_cfg = OmegaConf.create({
        "static": {
            "_target_": "calvin_env.camera.static_camera.StaticCamera",
            "name": "static",
            "fov": 10, "aspect": 1, "nearval": 0.01, "farval": 10,
            "width": 200, "height": 200,
            "look_at": [-0.026242, -0.030233, 0.392000],
            "look_from": [2.871459, -2.166602, 2.555160],
            "up_vector": [0.404140, 0.226298, 0.886262],
        },
        "gripper": {
            "_target_": "calvin_env.camera.gripper_camera.GripperCamera",
            "name": "gripper",
            "fov": 75, "aspect": 1, "nearval": 0.01, "farval": 2,
            "width": 84, "height": 84,
        },
    })

    full_cfg = OmegaConf.create({
        "data_path": CALVIN_ENV_DATA,
        "cameras": cam_cfg,
        "robot": OmegaConf.load('/home/kztrgg/calvin_env/conf/robot/panda.yaml'),
        "scene": OmegaConf.load('/home/kztrgg/calvin_env/conf/scene/calvin_scene_D.yaml'),
    })
    OmegaConf.resolve(full_cfg)

    from calvin_env.envs.play_table_env import PlayTableSimEnv
    env = PlayTableSimEnv(
        cameras=full_cfg.cameras,
        robot_cfg=full_cfg.robot,
        scene_cfg=full_cfg.scene,
        show_gui=False, use_vr=False, use_scene_info=True, use_egl=True,
        seed=seed, bullet_time_step=1.0/240, control_freq=30,
    )
    os.chdir(original_dir)
    return env


def get_tcp_pos(robot_obs):
    """Extract TCP position from robot_obs."""
    return robot_obs[:3]


def get_tcp_orn(robot_obs):
    """Extract TCP orientation (euler) from robot_obs."""
    return robot_obs[3:6]


def get_gripper_width(robot_obs):
    """Extract gripper opening width from robot_obs."""
    return robot_obs[6]


def make_rel_action(dx, dy, dz, droll=0, dpitch=0, dyaw=0, gripper=1.0):
    """Create a relative action.
    
    Actions in CALVIN are: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    Gripper: -1 = close, 1 = open
    Position deltas are scaled by 50 in the env, so small values suffice.
    """
    return np.array([dx, dy, dz, droll, dpitch, dyaw, gripper], dtype=np.float32)


# ═══════════════════════════════════════════════════
#  Scripted Policies for Individual Tasks
# ═══════════════════════════════════════════════════

class ScriptedPolicy:
    """Base class for scripted CALVIN task policies."""
    
    def __init__(self, task_name, language):
        self.task_name = task_name
        self.language = language
    
    def get_action(self, obs, step):
        raise NotImplementedError
    
    def max_steps(self):
        return 360


class MoveSliderPolicy(ScriptedPolicy):
    """Move slider left or right."""
    
    def __init__(self, direction='left'):
        lang = f"move the slider to the {direction}"
        super().__init__(f'move_slider_{direction}', lang)
        # Slider is at the top of the table
        self.direction = 1.0 if direction == 'left' else -1.0
        # Slider approximate position
        self.slider_y = 0.08  # y position of slider handle
        self.slider_z = 0.46  # table height
    
    def get_action(self, obs, step):
        tcp = get_tcp_pos(obs['robot_obs'])
        
        if step < 40:
            # Phase 1: Move above slider handle
            target_x = -0.10 * self.direction  # Center of slider
            target_y = self.slider_y
            target_z = self.slider_z + 0.15
            dx = np.clip((target_x - tcp[0]) * 3, -0.3, 0.3)
            dy = np.clip((target_y - tcp[1]) * 3, -0.3, 0.3)
            dz = np.clip((target_z - tcp[2]) * 3, -0.3, 0.3)
            return make_rel_action(dx, dy, dz, gripper=-1.0)
        
        elif step < 80:
            # Phase 2: Lower to slider height
            target_z = self.slider_z + 0.03
            dz = np.clip((target_z - tcp[2]) * 3, -0.3, 0.3)
            return make_rel_action(0, 0, dz, gripper=-1.0)
        
        else:
            # Phase 3: Push slider in direction
            dx = 0.15 * self.direction
            return make_rel_action(dx, 0, 0, gripper=-1.0)


class DrawerPolicy(ScriptedPolicy):
    """Open or close the drawer."""
    
    def __init__(self, action='open'):
        lang = f"{action} the drawer"
        super().__init__(f'{action}_drawer', lang)
        self.sign = 1.0 if action == 'open' else -1.0
        # Drawer handle approximate position
        self.drawer_x = 0.0
        self.drawer_y = -0.16
        self.drawer_z = 0.36
    
    def get_action(self, obs, step):
        tcp = get_tcp_pos(obs['robot_obs'])
        
        if step < 40:
            # Phase 1: Move to front of drawer handle
            target_x = self.drawer_x
            target_y = self.drawer_y + 0.05 * self.sign
            target_z = self.drawer_z + 0.1
            dx = np.clip((target_x - tcp[0]) * 3, -0.3, 0.3)
            dy = np.clip((target_y - tcp[1]) * 3, -0.3, 0.3)
            dz = np.clip((target_z - tcp[2]) * 3, -0.3, 0.3)
            return make_rel_action(dx, dy, dz, gripper=1.0)
        
        elif step < 80:
            # Phase 2: Lower and close gripper on handle
            target_z = self.drawer_z
            dz = np.clip((target_z - tcp[2]) * 3, -0.3, 0.3)
            return make_rel_action(0, 0, dz, gripper=-1.0)
        
        else:
            # Phase 3: Pull/push drawer
            dy = -0.15 * self.sign
            return make_rel_action(0, dy, 0, gripper=-1.0)


class LightPolicy(ScriptedPolicy):
    """Turn on/off lightbulb or LED."""
    
    def __init__(self, light='lightbulb', action='on'):
        lang = f"turn {'on' if action == 'on' else 'off'} the {light}"
        task = f"turn_{'on' if action == 'on' else 'off'}_{light}"
        super().__init__(task, lang)
        
        # Button/switch approximate positions
        if light == 'led':
            # LED is controlled by button
            self.target_x = -0.26
            self.target_y = -0.05
            self.target_z = 0.53
            self.is_button = True
        else:
            # Lightbulb is controlled by switch
            self.target_x = 0.18
            self.target_y = 0.03
            self.target_z = 0.53
            self.is_button = False
    
    def get_action(self, obs, step):
        tcp = get_tcp_pos(obs['robot_obs'])
        
        if step < 60:
            # Phase 1: Move above target
            target_z = self.target_z + 0.1
            dx = np.clip((self.target_x - tcp[0]) * 3, -0.3, 0.3)
            dy = np.clip((self.target_y - tcp[1]) * 3, -0.3, 0.3)
            dz = np.clip((target_z - tcp[2]) * 3, -0.3, 0.3)
            return make_rel_action(dx, dy, dz, gripper=-1.0)
        
        elif step < 120:
            # Phase 2: Press down
            dz = -0.08
            return make_rel_action(0, 0, dz, gripper=-1.0)
        
        else:
            # Phase 3: Lift back up
            dz = 0.05
            return make_rel_action(0, 0, dz, gripper=-1.0)


class PushBlockPolicy(ScriptedPolicy):
    """Push a block in a direction."""
    
    def __init__(self, block='red', direction='right'):
        lang = f"push the {block} block to the {direction}"
        task = f"push_{block}_block_{direction}"
        super().__init__(task, lang)
        self.direction = 1.0 if direction == 'right' else -1.0
    
    def get_action(self, obs, step):
        tcp = get_tcp_pos(obs['robot_obs'])
        scene = obs['scene_obs']
        
        # Block positions are in scene_obs indices:
        # red: 6:12, blue: 12:18, pink: 18:24
        block_pos = scene[6:9]  # red block position (x, y, z)
        
        if step < 60:
            # Phase 1: Move behind the block (opposite of push direction)
            target_x = block_pos[0] - 0.05 * self.direction
            target_y = block_pos[1]
            target_z = block_pos[2] + 0.06
            dx = np.clip((target_x - tcp[0]) * 3, -0.3, 0.3)
            dy = np.clip((target_y - tcp[1]) * 3, -0.3, 0.3)
            dz = np.clip((target_z - tcp[2]) * 3, -0.3, 0.3)
            return make_rel_action(dx, dy, dz, gripper=-1.0)
        
        elif step < 100:
            # Phase 2: Lower to block level
            target_z = block_pos[2] + 0.01
            dz = np.clip((target_z - tcp[2]) * 3, -0.3, 0.3)
            return make_rel_action(0, 0, dz, gripper=-1.0)
        
        else:
            # Phase 3: Push in direction
            dx = 0.1 * self.direction
            return make_rel_action(dx, 0, 0, gripper=-1.0)


class LiftBlockPolicy(ScriptedPolicy):
    """Lift a block from the table."""
    
    def __init__(self, block='red'):
        lang = f"lift the {block} block from the table"  
        task = f"lift_{block}_block_table"
        super().__init__(task, lang)
    
    def get_action(self, obs, step):
        tcp = get_tcp_pos(obs['robot_obs'])
        scene = obs['scene_obs']
        block_pos = scene[6:9]  # red block
        
        if step < 50:
            # Phase 1: Move above block
            target_x = block_pos[0]
            target_y = block_pos[1]
            target_z = block_pos[2] + 0.12
            dx = np.clip((target_x - tcp[0]) * 3, -0.3, 0.3)
            dy = np.clip((target_y - tcp[1]) * 3, -0.3, 0.3)
            dz = np.clip((target_z - tcp[2]) * 3, -0.3, 0.3)
            return make_rel_action(dx, dy, dz, gripper=1.0)
        
        elif step < 90:
            # Phase 2: Lower to grasp position
            target_z = block_pos[2] + 0.02
            dz = np.clip((target_z - tcp[2]) * 3, -0.3, 0.3)
            return make_rel_action(0, 0, dz, gripper=1.0)
        
        elif step < 120:
            # Phase 3: Close gripper
            return make_rel_action(0, 0, 0, gripper=-1.0)
        
        else:
            # Phase 4: Lift
            return make_rel_action(0, 0, 0.1, gripper=-1.0)


# ═══════════════════════════════════════════════════
#  All Policies
# ═══════════════════════════════════════════════════
ALL_POLICIES = [
    MoveSliderPolicy('left'),
    MoveSliderPolicy('right'),
    DrawerPolicy('open'),
    DrawerPolicy('close'),
    LightPolicy('lightbulb', 'on'),
    LightPolicy('lightbulb', 'off'),
    LightPolicy('led', 'on'),
    LightPolicy('led', 'off'),
    PushBlockPolicy('red', 'right'),
    LiftBlockPolicy('red'),
]


# ═══════════════════════════════════════════════════
#  Demo Generation 
# ═══════════════════════════════════════════════════
def generate_demos(output_dir, n_demos_per_task=50, max_steps=360, seed=0):
    """Generate demo data for all tasks.
    
    Output format matches CALVIN dataset:
    - episode_XXXXXXX.npz files with keys: rgb_static, rgb_gripper, robot_obs, 
      scene_obs, actions, rel_actions
    - Also saves lang_annotations/auto_lang_ann.npy with language and indices
    """
    output_dir = Path(output_dir)
    train_dir = output_dir / "training"
    val_dir = output_dir / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    global_step = 0
    
    # Storage for language annotations
    train_annotations = {
        'language': {'ann': [], 'task': [], 'emb': []},
        'info': {'indx': []},
    }
    val_annotations = {
        'language': {'ann': [], 'task': [], 'emb': []},
        'info': {'indx': []},
    }

    n_train = int(n_demos_per_task * 0.8)
    n_val = n_demos_per_task - n_train

    # Create a single environment instance (PyBullet doesn't handle re-creation well)
    env = make_env(seed=seed)

    for policy in ALL_POLICIES:
        print(f"\nTask: {policy.task_name} ({policy.language})")
        print(f"  Generating {n_train} train + {n_val} val demos...")
        
        for demo_i in range(n_demos_per_task):
            is_val = demo_i >= n_train
            target_dir = val_dir if is_val else train_dir
            annotations = val_annotations if is_val else train_annotations
            
            obs = env.reset()
            
            start_step = global_step
            
            for step in range(max_steps):
                action = policy.get_action(obs, step)
                
                # Save timestep data
                static_img = obs['rgb_obs']['rgb_static'].astype(np.uint8)
                gripper_img = obs['rgb_obs']['rgb_gripper'].astype(np.uint8)
                robot_obs_data = obs['robot_obs'].astype(np.float32)
                scene_obs_data = obs['scene_obs'].astype(np.float32)
                
                filename = f"episode_{global_step:07d}.npz"
                np.savez_compressed(
                    target_dir / filename,
                    rgb_static=static_img,
                    rgb_gripper=gripper_img,
                    robot_obs=robot_obs_data,
                    scene_obs=scene_obs_data,
                    rel_actions=action.astype(np.float32),
                    actions=action.astype(np.float32),
                )
                
                obs, reward, done, info = env.step(action)
                global_step += 1
                
                if done:
                    break
            
            end_step = global_step - 1
            
            # Store language annotation
            annotations['language']['ann'].append(policy.language)
            annotations['language']['task'].append(policy.task_name)
            annotations['language']['emb'].append(np.zeros(384))  # placeholder
            annotations['info']['indx'].append((start_step, end_step))
            
            if (demo_i + 1) % 10 == 0:
                print(f"    {demo_i + 1}/{n_demos_per_task} demos generated")
    
    env.close()
    
    # Save language annotations
    for annot, annot_dir in [(train_annotations, train_dir), (val_annotations, val_dir)]:
        lang_dir = annot_dir / "lang_annotations"
        lang_dir.mkdir(exist_ok=True)
        np.save(lang_dir / "auto_lang_ann.npy", annot, allow_pickle=True)
        print(f"\nSaved {len(annot['language']['ann'])} annotations to {lang_dir}")
    
    # Also save a simple episode index that maps (start_idx, end_idx) for each episode
    ep_lens_file = output_dir / "ep_lens.npy"
    all_indx = train_annotations['info']['indx'] + val_annotations['info']['indx']
    np.save(ep_lens_file, np.array(all_indx))
    
    print(f"\n{'='*60}")
    print(f"Generated {global_step} timesteps total")
    print(f"  Training: {len(train_annotations['language']['ann'])} episodes")
    print(f"  Validation: {len(val_annotations['language']['ann'])} episodes")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="/home/kztrgg/DynaCLIP/data/calvin/generated_demos")
    parser.add_argument("--n_demos", type=int, default=50,
                        help="Demos per task (80/20 train/val split)")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    generate_demos(args.output_dir, n_demos_per_task=args.n_demos,
                   max_steps=args.max_steps, seed=args.seed)
