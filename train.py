import argparse
from managers.trainer import *

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--config', type=str, default='./configs/debug.yaml')
args = parser.parse_args()
trainer = Trainer(args)
trainer.train()