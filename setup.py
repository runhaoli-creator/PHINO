from setuptools import setup, find_packages

setup(
    name="dynaclip",
    version="1.0.0",
    description="DynaCLIP: Physics-Grounded Visual Representations via Dynamics Contrastive Learning",
    author="Zhengtao Yao",
    url="https://github.com/zhengtaoyao/DynaCLIP",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "timm>=1.0.3",
        "transformers>=4.40.0",
        "tslearn>=0.6.3",
        "einops>=0.7.0",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "wandb>=0.16.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
        "matplotlib>=3.8.0",
        "scikit-learn>=1.4.0",
    ],
)
