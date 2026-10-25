"""Run and verify the shared educational analytics experiments."""
from pathlib import Path
import argparse

from experiments.run import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['smoke', 'measure'])
    parser.add_argument('--out', type=Path, help='New output directory.')
    parser.add_argument('--connection-file', type=Path, help='Private CLICKHOUSE_* environment file.')
    args = parser.parse_args()
    output = run_experiment(mode=args.action, out=args.out, connection_file=args.connection_file)
    print('OUTPUT', output)


if __name__ == '__main__':
    main()
