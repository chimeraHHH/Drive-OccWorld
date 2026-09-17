"""Run CPU memory contracts before the actual CUDA task-update profile."""
import argparse, subprocess, sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--train-cache',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    package=Path(__file__).resolve().parent
    subprocess.run([sys.executable,'-B',str(package/'test_observation_memory.py')],check=True)
    subprocess.run([sys.executable,'-B',str(package/'memory_experiment.py'),'--mode','preflight',
                    '--config',a.config,'--checkpoint',a.checkpoint,'--train-cache',a.train_cache,
                    '--out',a.out],check=True)


if __name__=='__main__':
    main()
