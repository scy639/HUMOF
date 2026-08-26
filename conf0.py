


import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--train', action='store_true')
_parser.add_argument('dataset', type=str, nargs='?', choices=['hik', 'hoi', 'gta', 'humanise'])
ARGS = _parser.parse_args()

DATASET_name:str = ARGS.dataset
# DATASET_name:str = 'hik'
# DATASET_name:str = 'hoi'
# DATASET_name:str = 'gta'
# DATASET_name:str = 'humanise'

TRAIN:bool = ARGS.train



CHECK_:bool = 1 # just some runtime check, disable it is ok


