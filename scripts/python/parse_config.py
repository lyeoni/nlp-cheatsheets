import json
from pathlib import Path

def parse_config(parser, config_path, root=Path('config')):
    _config_path = root/config_path
    with open(_config_path) as f:
        configs = json.load(f)

    for k, v in configs.items():
        if type(v) == dict:
            parser.set_defaults(**v)
        else:
            parser.set_defaults(**k)

    return parser

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_ds')

    # Parse saved configuration to arguments
    parser = parse_config(parser, 'sample.json', root=Path('../../sample'))

    args = parser.parse_args()
    print(args)
