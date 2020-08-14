import argparse
from typing import Optional, List
from pt_mlagents.trainers.learn import run_cli
from pt_mlagents.trainers.settings import RunOptions
from pt_mlagents.trainers.cli_utils import load_config


def parse_command_line(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("experiment_config_path")
    return parser.parse_args(argv)


def main():
    """
    Provides an alternative CLI interface to pt_mlagents-learn, 'pt_mlagents-run-experiment'.
    Accepts a JSON/YAML formatted pt_mlagents.trainers.learn.RunOptions object, and executes
    the run loop as defined in pt_mlagents.trainers.learn.run_cli.
    """
    args = parse_command_line()
    expt_config = load_config(args.experiment_config_path)
    run_cli(RunOptions.from_dict(expt_config))


if __name__ == "__main__":
    main()
