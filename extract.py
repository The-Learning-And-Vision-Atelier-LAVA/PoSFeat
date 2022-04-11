import argparse
from managers.extractor import *

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--config', type=str, default='./configs/extract.yaml')
args = parser.parse_args()
extractor = Extractor(args)
extractor.extract()