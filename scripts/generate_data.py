"""
DynaCLIP Data Generation Script.

Generates category-grounded physics data using analytical physics engine
paired with real images from DomainNet/COCO.
"""

import logging
import time
import sys

sys.path.insert(0, ".")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/data_generation.log"),
    ],
)

from dynaclip.data.generation import DynaCLIPDataGenerator

logger = logging.getLogger(__name__)


def main():
    t0 = time.time()
    logger.info("Starting DynaCLIP data generation (category-grounded physics)")

    gen = DynaCLIPDataGenerator(
        output_dir="data_cache/dynaclip_data",
        dataset_root="/home/kztrgg/datasets",
        num_physics_per_image=5,
        max_images=20000,
        seed=42,
        similarity_metric="l2",  # l2 is much faster than DTW
    )
    gen.generate_all(compute_similarity=True, max_sim_pairs=500000)

    elapsed = (time.time() - t0) / 60
    logger.info(f"Data generation completed in {elapsed:.1f} minutes")
    print(f"\nDone in {elapsed:.1f} min")


if __name__ == "__main__":
    main()
