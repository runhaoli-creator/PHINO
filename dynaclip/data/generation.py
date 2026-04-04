"""
DynaCLIP Data Generation: Category-Grounded Physics Simulation.

Assigns physics properties (mass, friction, restitution) based on the MATERIAL
category of each object (metal, wood, fabric, glass, rubber, food, paper, animal).
This creates a learnable correlation between visual appearance and physics, which
is the core of DynaCLIP's pre-training signal.

Uses an analytical physics engine to generate dynamics fingerprints, paired with
REAL images from DomainNet (345 categories) and COCO datasets.
"""

import json
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

logger = logging.getLogger(__name__)


# ==========================================================================
# Material-based physics priors for DomainNet categories
# ==========================================================================
MATERIAL_PRIORS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "metal": {
        "mass": (2.0, 8.0),
        "static_friction": (0.10, 0.40),
        "restitution": (0.50, 0.90),
    },
    "wood": {
        "mass": (0.5, 3.0),
        "static_friction": (0.30, 0.60),
        "restitution": (0.20, 0.40),
    },
    "fabric": {
        "mass": (0.05, 0.50),
        "static_friction": (0.60, 1.20),
        "restitution": (0.00, 0.15),
    },
    "glass_ceramic": {
        "mass": (0.30, 2.00),
        "static_friction": (0.10, 0.30),
        "restitution": (0.40, 0.70),
    },
    "rubber_plastic": {
        "mass": (0.20, 1.50),
        "static_friction": (0.50, 1.00),
        "restitution": (0.30, 0.65),
    },
    "food_organic": {
        "mass": (0.05, 1.00),
        "static_friction": (0.30, 0.60),
        "restitution": (0.00, 0.20),
    },
    "paper_light": {
        "mass": (0.01, 0.20),
        "static_friction": (0.30, 0.50),
        "restitution": (0.00, 0.10),
    },
    "animal": {
        "mass": (0.50, 5.00),
        "static_friction": (0.40, 0.80),
        "restitution": (0.05, 0.25),
    },
    "stone_heavy": {
        "mass": (3.0, 10.0),
        "static_friction": (0.40, 0.70),
        "restitution": (0.15, 0.35),
    },
    "default": {
        "mass": (0.30, 2.00),
        "static_friction": (0.30, 0.60),
        "restitution": (0.15, 0.40),
    },
}

CATEGORY_TO_MATERIAL: Dict[str, str] = {
    # Metal
    "anvil": "metal", "axe": "metal", "calculator": "metal", "cannon": "metal",
    "clarinet": "metal", "compass": "metal", "drill": "metal",
    "dumbbell": "metal", "flashlight": "metal", "fork": "metal",
    "frying_pan": "metal", "hammer": "metal", "helmet": "metal",
    "key": "metal", "knife": "metal", "lighter": "metal",
    "nail": "metal", "pliers": "metal", "rifle": "metal",
    "saw": "metal", "scissors": "metal", "screwdriver": "metal",
    "shovel": "metal", "sword": "metal", "toaster": "metal",
    "trombone": "metal", "trumpet": "metal", "fire_hydrant": "metal",
    "wristwatch": "metal", "crown": "metal", "stove": "metal",
    "oven": "metal", "sink": "metal", "washing_machine": "metal",
    "dishwasher": "metal", "microwave": "metal", "traffic_light": "metal",
    "streetlight": "metal", "power_outlet": "metal", "stop_sign": "metal",
    "spoon": "metal", "chandelier": "metal", "alarm_clock": "metal",
    "saxophone": "metal", "harp": "metal", "golf_club": "metal",
    "hockey_stick": "metal", "rake": "metal", "syringe": "metal",
    "stethoscope": "metal", "stereo": "metal",
    # Wood
    "baseball_bat": "wood", "bench": "wood", "canoe": "wood",
    "chair": "wood", "door": "wood", "fence": "wood",
    "guitar": "wood", "ladder": "wood", "piano": "wood",
    "table": "wood", "tree": "wood", "pencil": "wood",
    "drums": "wood", "diving_board": "wood", "picture_frame": "wood",
    "dresser": "wood", "floor_lamp": "wood", "barn": "wood",
    "bridge": "wood", "stairs": "wood", "broom": "wood",
    "swing_set": "wood", "see_saw": "wood", "windmill": "wood",
    "violin": "wood", "cello": "wood", "basket": "wood",
    "boomerang": "wood",
    # Fabric
    "bandage": "fabric", "bed": "fabric", "belt": "fabric",
    "bowtie": "fabric", "couch": "fabric", "hat": "fabric",
    "jacket": "fabric", "pants": "fabric", "pillow": "fabric",
    "shoe": "fabric", "shorts": "fabric", "sleeping_bag": "fabric",
    "sock": "fabric", "sweater": "fabric", "t-shirt": "fabric",
    "tent": "fabric", "umbrella": "fabric", "parachute": "fabric",
    "purse": "fabric", "suitcase": "fabric", "backpack": "fabric",
    "yoga": "fabric", "flip_flops": "fabric", "stitches": "fabric",
    "underwear": "fabric", "necklace": "fabric",
    # Glass / Ceramic
    "bathtub": "glass_ceramic", "cup": "glass_ceramic",
    "light_bulb": "glass_ceramic", "mug": "glass_ceramic",
    "vase": "glass_ceramic", "wine_bottle": "glass_ceramic",
    "wine_glass": "glass_ceramic", "coffee_cup": "glass_ceramic",
    "teapot": "glass_ceramic", "hourglass": "glass_ceramic",
    "lantern": "glass_ceramic", "bottlecap": "glass_ceramic",
    "toilet": "glass_ceramic",
    # Rubber / Plastic
    "airplane": "rubber_plastic", "basketball": "rubber_plastic",
    "bicycle": "rubber_plastic", "bus": "rubber_plastic",
    "car": "rubber_plastic", "cell_phone": "rubber_plastic",
    "computer": "rubber_plastic", "fan": "rubber_plastic",
    "keyboard": "rubber_plastic", "laptop": "rubber_plastic",
    "mouse": "rubber_plastic", "remote_control": "rubber_plastic",
    "submarine": "rubber_plastic", "telephone": "rubber_plastic",
    "television": "rubber_plastic", "toothbrush": "rubber_plastic",
    "motorbike": "rubber_plastic", "rollerskates": "rubber_plastic",
    "skateboard": "rubber_plastic", "hockey_puck": "rubber_plastic",
    "soccer_ball": "rubber_plastic", "firetruck": "rubber_plastic",
    "school_bus": "rubber_plastic", "truck": "rubber_plastic",
    "van": "rubber_plastic", "helicopter": "rubber_plastic",
    "hot_air_balloon": "rubber_plastic", "sailboat": "rubber_plastic",
    "speedboat": "rubber_plastic", "tractor": "rubber_plastic",
    "pickup_truck": "rubber_plastic", "cruise_ship": "rubber_plastic",
    "train": "rubber_plastic", "ambulance": "rubber_plastic",
    "police_car": "rubber_plastic", "flying_saucer": "rubber_plastic",
    "eraser": "rubber_plastic", "crayon": "rubber_plastic",
    "marker": "rubber_plastic", "radio": "rubber_plastic",
    "headphones": "rubber_plastic", "binoculars": "rubber_plastic",
    "camera": "rubber_plastic", "baseball": "rubber_plastic",
    "snorkel": "rubber_plastic", "megaphone": "rubber_plastic",
    "microphone": "rubber_plastic", "tennis_racquet": "rubber_plastic",
    "paintbrush": "rubber_plastic", "paint_can": "rubber_plastic",
    "ceiling_fan": "rubber_plastic", "garden_hose": "rubber_plastic",
    "wheel": "rubber_plastic", "aircraft_carrier": "rubber_plastic",
    "bulldozer": "rubber_plastic", "cooler": "rubber_plastic",
    "toothpaste": "rubber_plastic", "passport": "rubber_plastic",
    "paper_clip": "rubber_plastic", "mailbox": "rubber_plastic",
    "eyeglasses": "rubber_plastic",
    # Food / Organic
    "apple": "food_organic", "banana": "food_organic",
    "birthday_cake": "food_organic", "blueberry": "food_organic",
    "bread": "food_organic", "broccoli": "food_organic",
    "cake": "food_organic", "cookie": "food_organic",
    "donut": "food_organic", "grapes": "food_organic",
    "hamburger": "food_organic", "hot_dog": "food_organic",
    "ice_cream": "food_organic", "lollipop": "food_organic",
    "mushroom": "food_organic", "onion": "food_organic",
    "peanut": "food_organic", "pear": "food_organic",
    "pineapple": "food_organic", "pizza": "food_organic",
    "potato": "food_organic", "sandwich": "food_organic",
    "steak": "food_organic", "strawberry": "food_organic",
    "watermelon": "food_organic", "asparagus": "food_organic",
    "blackberry": "food_organic", "carrot": "food_organic",
    "peas": "food_organic", "string_bean": "food_organic",
    "popsicle": "food_organic",
    # Paper / Light
    "book": "paper_light", "calendar": "paper_light",
    "envelope": "paper_light", "map": "paper_light",
    "postcard": "paper_light", "spreadsheet": "paper_light",
    "leaf": "paper_light", "feather": "paper_light",
    "flower": "paper_light", "candle": "paper_light",
    "matches": "paper_light", "snowflake": "paper_light",
    "lipstick": "paper_light", "bracelet": "paper_light",
    # Animal
    "ant": "animal", "bat": "animal", "bear": "animal",
    "bird": "animal", "butterfly": "animal", "camel": "animal",
    "cat": "animal", "cow": "animal", "crab": "animal",
    "crocodile": "animal", "dog": "animal", "dolphin": "animal",
    "dragon": "animal", "duck": "animal", "elephant": "animal",
    "fish": "animal", "flamingo": "animal", "frog": "animal",
    "giraffe": "animal", "hedgehog": "animal", "horse": "animal",
    "kangaroo": "animal", "lion": "animal", "lobster": "animal",
    "monkey": "animal", "mosquito": "animal", "octopus": "animal",
    "owl": "animal", "panda": "animal", "parrot": "animal",
    "penguin": "animal", "pig": "animal", "rabbit": "animal",
    "raccoon": "animal", "rhinoceros": "animal", "scorpion": "animal",
    "sea_turtle": "animal", "shark": "animal", "sheep": "animal",
    "snail": "animal", "snake": "animal", "spider": "animal",
    "squirrel": "animal", "swan": "animal", "tiger": "animal",
    "whale": "animal", "zebra": "animal", "bee": "animal",
    "teddy-bear": "animal", "mermaid": "animal", "angel": "animal",
    "animal_migration": "animal",
    # Stone / Heavy
    "castle": "stone_heavy", "church": "stone_heavy",
    "hospital": "stone_heavy", "house": "stone_heavy",
    "jail": "stone_heavy", "lighthouse": "stone_heavy",
    "The_Eiffel_Tower": "stone_heavy", "The_Great_Wall_of_China": "stone_heavy",
    "skyscraper": "stone_heavy", "fireplace": "stone_heavy",
    "hot_tub": "stone_heavy", "roller_coaster": "stone_heavy",
    "waterslide": "stone_heavy",
    # Default
    "arm": "default", "ear": "default", "elbow": "default",
    "eye": "default", "face": "default", "finger": "default",
    "foot": "default", "goatee": "default", "hand": "default",
    "knee": "default", "leg": "default", "mouth": "default",
    "moustache": "default", "nose": "default", "tooth": "default",
    "toe": "default", "brain": "default", "skull": "default",
    "smiley_face": "default",
    "circle": "default", "diamond": "default", "hexagon": "default",
    "line": "default", "octagon": "default", "square": "default",
    "squiggle": "default", "triangle": "default", "zigzag": "default",
    "star": "default",
    "beach": "default", "bush": "default", "cactus": "default",
    "campfire": "default", "cloud": "default", "grass": "default",
    "hurricane": "default", "lightning": "default", "moon": "default",
    "mountain": "default", "ocean": "default", "palm_tree": "default",
    "pond": "default", "pool": "default", "rain": "default",
    "rainbow": "default", "river": "default", "snowman": "default",
    "sun": "default", "tornado": "default", "garden": "default",
    "house_plant": "default", "camouflage": "default",
    "The_Mona_Lisa": "default", "beard": "default",
    "coco_object": "default",
}


def get_material_for_category(category: str) -> str:
    return CATEGORY_TO_MATERIAL.get(category, "default")


# ==========================================================================
# Multi-material mapping (v2): categories can map to multiple materials
# with probability weights. This breaks the deterministic
# category→material→physics confound identified in the P0-1 critique.
# ==========================================================================
CATEGORY_TO_MATERIALS_PROB: Dict[str, Dict[str, float]] = {}

def _build_multi_material_mapping():
    """Build probabilistic multi-material mapping from base 1-to-1 mapping.

    Rules:
    1. Primary material gets 50% probability
    2. 2-3 secondary materials get remaining 50%, weighted by physical plausibility
    3. Every category can potentially have any material via a small "surprise" weight
    """
    # Define material compatibility groups (materials that are physically plausible
    # alternatives for objects typically made of the primary material)
    MATERIAL_ALTERNATES = {
        "metal": ["rubber_plastic", "glass_ceramic", "wood"],
        "wood": ["rubber_plastic", "metal", "paper_light"],
        "fabric": ["rubber_plastic", "paper_light", "animal"],
        "glass_ceramic": ["metal", "rubber_plastic", "stone_heavy"],
        "rubber_plastic": ["metal", "glass_ceramic", "wood"],
        "food_organic": ["rubber_plastic", "paper_light", "fabric"],
        "paper_light": ["fabric", "food_organic", "rubber_plastic"],
        "animal": ["rubber_plastic", "fabric", "food_organic"],
        "stone_heavy": ["metal", "glass_ceramic", "wood"],
        "default": ["rubber_plastic", "metal", "wood"],
    }

    for category, primary_material in CATEGORY_TO_MATERIAL.items():
        alternates = MATERIAL_ALTERNATES.get(primary_material, ["default"])
        prob_map = {primary_material: 0.50}
        remaining = 0.50
        n_alt = len(alternates)
        for i, alt in enumerate(alternates):
            # Decreasing weight for less likely alternates
            w = remaining * (0.5 ** (i + 1)) if i < n_alt - 1 else remaining
            prob_map[alt] = w
            remaining -= w
        CATEGORY_TO_MATERIALS_PROB[category] = prob_map

_build_multi_material_mapping()


def sample_material_for_category(category: str, rng: np.random.Generator) -> str:
    """Sample a material type probabilistically for a given category.

    Unlike get_material_for_category (deterministic 1-to-1), this allows
    the same category to produce different material types across samples,
    breaking the category→physics confound.
    """
    prob_map = CATEGORY_TO_MATERIALS_PROB.get(
        category, {"default": 1.0}
    )
    materials = list(prob_map.keys())
    probs = np.array(list(prob_map.values()))
    probs = probs / probs.sum()  # normalize
    return str(rng.choice(materials, p=probs))


# ==========================================================================
# Physical property configuration
# ==========================================================================
@dataclass
class PhysicsConfig:
    mass: float = 1.0
    static_friction: float = 0.5
    dynamic_friction: float = 0.4
    restitution: float = 0.3
    material: str = "default"

    def __post_init__(self):
        self.dynamic_friction = 0.8 * self.static_friction

    @staticmethod
    def sample_for_category(category: str, rng: np.random.Generator,
                            multi_material: bool = True) -> "PhysicsConfig":
        """Sample physics from category-appropriate material prior.

        If multi_material=True (v2), samples material probabilistically
        to break the deterministic category→physics confound.
        If multi_material=False (legacy), uses the 1-to-1 mapping.
        """
        if multi_material:
            material = sample_material_for_category(category, rng)
        else:
            material = get_material_for_category(category)
        prior = MATERIAL_PRIORS[material]
        mass = rng.uniform(*prior["mass"])
        sf = rng.uniform(*prior["static_friction"])
        rest = rng.uniform(*prior["restitution"])
        return PhysicsConfig(mass=float(mass), static_friction=float(sf),
                             restitution=float(rest), material=material)

    @staticmethod
    def sample_random(rng: np.random.Generator) -> "PhysicsConfig":
        """Legacy: uniform random sampling (no category correlation)."""
        mass = np.exp(rng.uniform(np.log(0.05), np.log(10.0)))
        static_friction = rng.uniform(0.05, 1.5)
        restitution = rng.uniform(0.0, 0.95)
        return PhysicsConfig(mass=float(mass), static_friction=float(static_friction),
                             restitution=float(restitution), material="random")

    @property
    def uid(self) -> str:
        s = f"m{self.mass:.6f}_sf{self.static_friction:.6f}_r{self.restitution:.6f}"
        return hashlib.md5(s.encode()).hexdigest()[:12]

    def to_vector(self) -> np.ndarray:
        return np.array([np.log(self.mass + 1e-6) / np.log(10.0),
                         self.static_friction / 1.5,
                         self.restitution / 0.95], dtype=np.float32)


# ==========================================================================
# Diagnostic Actions
# ==========================================================================
@dataclass
class DiagnosticAction:
    name: str
    action_type: str
    velocity: float = 0.05
    duration: float = 1.0
    direction: np.ndarray = field(
        default_factory=lambda: np.array([1, 0, 0], dtype=np.float64)
    )


DIAGNOSTIC_ACTIONS = [
    DiagnosticAction("push_x", "push", 0.05, 1.0, np.array([1., 0., 0.])),
    DiagnosticAction("push_y", "push", 0.05, 1.0, np.array([0., 1., 0.])),
    DiagnosticAction("grasp_lift_release", "grasp_lift", 0.02, 2.0, np.array([0., 0., 1.])),
    DiagnosticAction("lateral_flick", "flick", 0.3, 0.2, np.array([1., 0.5, 0.])),
    DiagnosticAction("slow_press_down", "press", 0.02, 1.5, np.array([0., 0., -1.])),
]


@dataclass
class TrajectoryRecord:
    action_name: str
    timesteps: int = 50
    hz: int = 20
    data: np.ndarray = field(default_factory=lambda: np.zeros((50, 13)))


# ==========================================================================
# Analytical Physics Engine
# ==========================================================================
class AnalyticalPhysicsEngine:
    """Generates physics-dependent trajectories via Newtonian simulation."""
    GRAVITY = np.array([0., 0., -9.81])
    DT = 1.0 / 20.0

    def execute_diagnostic_action(self, action: DiagnosticAction,
                                  physics: PhysicsConfig, timesteps: int = 50) -> np.ndarray:
        traj = np.zeros((timesteps, 13))
        pos, vel, ang_vel = np.zeros(3), np.zeros(3), np.zeros(3)
        quat = np.array([1., 0., 0., 0.])
        mass, mu_s, mu_d, rest = physics.mass, physics.static_friction, physics.dynamic_friction, physics.restitution
        for t in range(timesteps):
            t_frac = t / timesteps
            af = np.zeros(3)
            at = np.zeros(3)
            if action.action_type == "push":
                if t_frac < action.duration / (timesteps * self.DT):
                    af = action.direction * action.velocity * 50.0
                    at = np.cross(np.array([0., 0., 0.02]), af) * 0.1
            elif action.action_type == "grasp_lift":
                if t_frac < 0.3:
                    af = action.direction * 15.0
                elif t_frac < 0.6:
                    af = action.direction * mass * 9.81
            elif action.action_type == "flick":
                if t < 4:
                    af = action.direction * action.velocity * 200.0
                    at = np.array([0., 0., action.velocity * 5.0])
            elif action.action_type == "press":
                if t_frac < action.duration / (timesteps * self.DT):
                    af = action.direction * action.velocity * 30.0
            spd = np.linalg.norm(vel[:2])
            if spd > 1e-8:
                fd = -vel[:2] / spd
                fm = mu_d * mass * 9.81
                ff = np.array([fd[0]*fm, fd[1]*fm, 0.])
            elif np.linalg.norm(af[:2]) > mu_s * mass * 9.81:
                ff = np.zeros(3)
            else:
                ff = -af.copy(); ff[2] = 0.
            net = af + ff + self.GRAVITY * mass
            acc = net / mass
            vel += acc * self.DT
            pos += vel * self.DT
            if pos[2] < 0.:
                pos[2] = 0.
                if vel[2] < 0:
                    vel[2] = -vel[2] * rest
                    vel[:2] *= max(0, 1.0 - mu_d * 0.5)
            ang_acc = at / (mass * 0.01)
            ang_vel += ang_acc * self.DT
            ang_vel *= 0.98
            angle = np.linalg.norm(ang_vel) * self.DT
            if angle > 1e-8:
                axis = ang_vel / np.linalg.norm(ang_vel)
                dq = np.array([np.cos(angle/2), *(axis * np.sin(angle/2))])
                quat = _quat_mul(quat, dq)
                quat /= np.linalg.norm(quat)
            traj[t, :3] = pos
            traj[t, 3:7] = quat
            traj[t, 7:10] = vel
            traj[t, 10:13] = ang_vel
        return traj


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


# ==========================================================================
# Dynamics Similarity
# ==========================================================================
def compute_dynamics_similarity_dtw(traj1: np.ndarray, traj2: np.ndarray) -> float:
    try:
        from tslearn.metrics import dtw as tslearn_dtw
        d = tslearn_dtw(traj1, traj2)
        return float(np.exp(-d / 10.0))
    except ImportError:
        return compute_dynamics_similarity_l2(traj1, traj2)

def compute_dynamics_similarity_l2(traj1: np.ndarray, traj2: np.ndarray) -> float:
    diff = np.linalg.norm(traj1[-1] - traj2[-1])
    return float(np.exp(-diff / 5.0))

def compute_dynamics_similarity_mse(traj1: np.ndarray, traj2: np.ndarray) -> float:
    mse = np.mean((traj1 - traj2) ** 2)
    return float(np.exp(-mse / 5.0))

def compute_dynamics_similarity_velocity_dtw(traj1: np.ndarray, traj2: np.ndarray) -> float:
    try:
        from tslearn.metrics import dtw as tslearn_dtw
        d = tslearn_dtw(traj1[:, 7:13], traj2[:, 7:13])
        return float(np.exp(-d / 10.0))
    except ImportError:
        return compute_dynamics_similarity_l2(traj1, traj2)

SIMILARITY_METRICS = {
    "dtw": compute_dynamics_similarity_dtw,
    "l2": compute_dynamics_similarity_l2,
    "mse": compute_dynamics_similarity_mse,
    "velocity_dtw": compute_dynamics_similarity_velocity_dtw,
}


# ==========================================================================
# Image Source
# ==========================================================================
def collect_real_images(dataset_root: str = "/home/kztrgg/datasets",
                        max_images: int = 50000, seed: int = 42) -> List[Tuple[str, str]]:
    rng = np.random.default_rng(seed)
    images: List[Tuple[str, str]] = []
    domainnet_real = Path(dataset_root) / "domainnet" / "real"
    if domainnet_real.exists():
        for cat_dir in sorted(domainnet_real.iterdir()):
            if cat_dir.is_dir():
                for img_path in sorted(cat_dir.glob("*.jpg")):
                    images.append((str(img_path), cat_dir.name))
        logger.info(f"Found {len(images)} DomainNet real images")
    coco_dir = Path(dataset_root) / "coco" / "raw" / "train2017"
    if coco_dir.exists() and len(images) < max_images:
        coco_imgs = sorted(coco_dir.glob("*.jpg"))
        for img_path in coco_imgs[:max_images - len(images)]:
            images.append((str(img_path), "coco_object"))
    rng.shuffle(images)
    images = images[:max_images]
    logger.info(f"Collected {len(images)} real images total")
    return images


# ==========================================================================
# Full Data Generation Pipeline
# ==========================================================================
class DynaCLIPDataGenerator:
    def __init__(self, output_dir: str = "data_cache/dynaclip_data",
                 dataset_root: str = "/home/kztrgg/datasets",
                 num_physics_per_image: int = 5, max_images: int = 20000,
                 seed: int = 42, similarity_metric: str = "dtw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_root = dataset_root
        self.num_physics = num_physics_per_image
        self.max_images = max_images
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sim_fn = SIMILARITY_METRICS.get(similarity_metric, compute_dynamics_similarity_dtw)
        self.engine = AnalyticalPhysicsEngine()

    def generate_all(self, compute_similarity: bool = True, max_sim_pairs: int = 500000):
        logger.info("=== Starting DynaCLIP data generation (category-grounded) ===")
        images = collect_real_images(self.dataset_root, self.max_images, self.seed)

        mat_counts: Dict[str, int] = {}
        for _, cat in images:
            mat = get_material_for_category(cat)
            mat_counts[mat] = mat_counts.get(mat, 0) + 1
        logger.info(f"Material distribution: {mat_counts}")

        metadata = []
        fingerprints = []
        fp_dir = self.output_dir / "fingerprints"
        fp_dir.mkdir(parents=True, exist_ok=True)

        for img_idx, (img_path, category) in enumerate(images):
            if img_idx % 2000 == 0:
                logger.info(f"Processing image {img_idx}/{len(images)}")
            for phys_idx in range(self.num_physics):
                physics = PhysicsConfig.sample_for_category(category, self.rng)
                gidx = img_idx * self.num_physics + phys_idx
                all_traj = [self.engine.execute_diagnostic_action(a, physics) for a in DIAGNOSTIC_ACTIONS]
                flat = np.concatenate(all_traj, axis=0)
                fp_path = fp_dir / f"fp_{gidx:06d}.npz"
                np.savez_compressed(str(fp_path), flat_trajectory=flat,
                                    mass=physics.mass, static_friction=physics.static_friction,
                                    dynamic_friction=physics.dynamic_friction, restitution=physics.restitution)
                fingerprints.append(flat)
                metadata.append({
                    "image_path": img_path, "category": category,
                    "material": physics.material,
                    "global_idx": gidx, "image_group": img_idx,
                    "physics_uid": physics.uid,
                    "mass": float(physics.mass),
                    "static_friction": float(physics.static_friction),
                    "dynamic_friction": float(physics.dynamic_friction),
                    "restitution": float(physics.restitution),
                    "fingerprint_path": str(fp_path),
                })

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        logger.info(f"Saved metadata for {len(metadata)} entries")

        if compute_similarity:
            self._compute_similarity_matrix(fingerprints, max_pairs=max_sim_pairs)
        self._generate_invisible_physics_testset(images)
        self._generate_cross_material_testset(images)
        logger.info("=== Data generation complete ===")

    def _compute_similarity_matrix(self, fingerprints: list, max_pairs: int = 500000):
        n = len(fingerprints)
        num_pairs = min(max_pairs, n * (n - 1) // 2)
        logger.info(f"Computing {num_pairs} pairwise similarities for {n} configs")
        pairs = np.zeros((num_pairs, 2), dtype=np.int64)
        sims = np.zeros(num_pairs, dtype=np.float32)
        for k in range(num_pairs):
            if k % 100000 == 0:
                logger.info(f"  Pair {k}/{num_pairs}")
            i, j = self.rng.choice(n, size=2, replace=False)
            pairs[k] = [i, j]
            sims[k] = self.sim_fn(fingerprints[i], fingerprints[j])
        np.savez_compressed(self.output_dir / "similarity_matrix.npz", pairs=pairs, similarities=sims)
        logger.info(f"Saved {num_pairs} similarities (mean={sims.mean():.4f}, std={sims.std():.4f})")

    def _generate_invisible_physics_testset(self, images: list, num_pairs: int = 500):
        logger.info("Generating Invisible Physics test set")
        test_pairs = []
        for _ in range(num_pairs):
            idx = self.rng.integers(0, len(images))
            img_path, cat = images[idx]
            pa = PhysicsConfig.sample_for_category(cat, self.rng)
            pb = PhysicsConfig.sample_for_category(cat, self.rng)
            attempts = 0
            while abs(pa.mass - pb.mass) < 0.3 and attempts < 20:
                pb = PhysicsConfig.sample_for_category(cat, self.rng)
                attempts += 1
            fp_a = np.concatenate([self.engine.execute_diagnostic_action(a, pa) for a in DIAGNOSTIC_ACTIONS])
            fp_b = np.concatenate([self.engine.execute_diagnostic_action(a, pb) for a in DIAGNOSTIC_ACTIONS])
            dyn_sim = float(self.sim_fn(fp_a, fp_b))
            test_pairs.append({"image_path": img_path, "category": cat,
                               "material": pa.material,
                               "physics_a": asdict(pa), "physics_b": asdict(pb),
                               "dynamics_similarity": dyn_sim})
        with open(self.output_dir / "invisible_physics_test.json", "w") as f:
            json.dump(test_pairs, f, indent=2)
        logger.info(f"Saved {num_pairs} invisible physics test pairs")

    def _generate_cross_material_testset(self, images: list, num_pairs: int = 1000):
        """Pairs from different material clusters for cross-material evaluation."""
        logger.info("Generating Cross-Material test set")
        by_material: Dict[str, List[Tuple[str, str]]] = {}
        for img_path, cat in images:
            mat = get_material_for_category(cat)
            by_material.setdefault(mat, []).append((img_path, cat))
        materials = [m for m in by_material if len(by_material[m]) >= 10]
        test_pairs = []
        for _ in range(num_pairs):
            m1, m2 = self.rng.choice(materials, size=2, replace=False)
            img1, cat1 = by_material[m1][self.rng.integers(len(by_material[m1]))]
            img2, cat2 = by_material[m2][self.rng.integers(len(by_material[m2]))]
            p1 = PhysicsConfig.sample_for_category(cat1, self.rng)
            p2 = PhysicsConfig.sample_for_category(cat2, self.rng)
            fp1 = np.concatenate([self.engine.execute_diagnostic_action(a, p1) for a in DIAGNOSTIC_ACTIONS])
            fp2 = np.concatenate([self.engine.execute_diagnostic_action(a, p2) for a in DIAGNOSTIC_ACTIONS])
            dyn_sim = float(self.sim_fn(fp1, fp2))
            test_pairs.append({
                "image_path_a": img1, "category_a": cat1, "material_a": m1,
                "image_path_b": img2, "category_b": cat2, "material_b": m2,
                "physics_a": asdict(p1), "physics_b": asdict(p2),
                "dynamics_similarity": dyn_sim,
            })
        with open(self.output_dir / "cross_material_test.json", "w") as f:
            json.dump(test_pairs, f, indent=2)
        logger.info(f"Saved {num_pairs} cross-material test pairs")
