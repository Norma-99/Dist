import argparse
import logging
from differential_privacy.controller import Controller
from differential_privacy.utils import read_config


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    Controller(config).run()



