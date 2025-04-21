import subprocess
import sys
import importlib.util

PACKAGES = ['scikit-learn', 'numpy', 'matplotlib', 'torch', 'pytorch-lightning', 'wandb', 'einops', 'torchvision', 'pandas']

def install_packages(package = []):
    for pckg in PACKAGES + package:
        print()
        print(pckg)
        subprocess.run([sys.executable, '-m', 'pip', 'install', pckg, '-q'], check=True)
