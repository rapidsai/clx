import yaml

"""
Utility script
"""


def load_yaml(yaml_file):
    with open(yaml_file) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
