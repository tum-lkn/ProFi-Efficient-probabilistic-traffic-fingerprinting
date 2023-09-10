import json
import os

cache_path = '/opt/project/data/cache'


def is_cached(file_name) -> bool:
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    return os.path.exists(os.path.join(cache_path, file_name))


def add_to_cache(file_name, element) -> None:
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    with open(os.path.join(cache_path, file_name), 'w') as fh:
        json.dump(element, fh)


def read_cache(file_name) -> object:
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    with open(os.path.join(cache_path, file_name), 'r') as fh:
        return json.load(fh)

