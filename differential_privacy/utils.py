import json


def read_config(path:str):
    with open(path) as conf_file:
        return json.load(conf_file)
