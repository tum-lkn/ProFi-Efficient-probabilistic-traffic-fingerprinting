import implementation.classification.mc as mcmod
import argparse
import json
import os


def make_export(trial_dir: str, prefix: str) -> None:
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    model_path = os.path.join(trial_dir, 'model.json')
    config_path = os.path.join(trial_dir, 'config.json')
    # Get config and model from trial dir.
    with open(model_path, 'r') as fh:
        model_d = json.load(fh)
    with open(os.path.join(prefix, 'model.json'), 'w') as fh:
        json.dump(model_d, fh)
    # Save model and config in destination dir.
    with open(config_path, 'r') as fh:
        config_d = json.load(fh)
    with open(os.path.join(prefix, 'config.json'), 'w') as fh:
        json.dump(config_d, fh)
    # Create C-Export.
    mc = mcmod.MarkovChain.from_dict(model_d)
    export = mc.c_export()
    export['trace_length'] = config_d['seq_length']
    lbl = config_d['label'].replace('.', '_')
    with open(os.path.join(prefix, f'{lbl}_c_export.json'), 'w') as fh:
        json.dump(export, fh, indent=1)


def biggest_mc():
    """
    www.tahiamasr.com

    Returns:

    """
    prefix = '/opt/project/data/biggest-mc'
    trial_dir = "/opt/project/data/grid-search-results/72d6bfe89e11439395c0c6986d3507f1_0"
    make_export(trial_dir, prefix)


def smallest_mc():
    """
    www.grammarly.com

    Returns:

    """
    prefix = '/opt/project/data/smallest-mc'
    trial_dir = "/opt/project/data/grid-search-results/8a170bb98950452ab3327581521157fb_0"
    make_export(trial_dir, prefix)


if __name__ == '__main__':
    biggest_mc()
    smallest_mc()