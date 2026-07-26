"""Unified entry point for EMRB problem generation."""
import argparse
import sys


LEVEL_DEFAULTS = {
    'L1': {'seed_start': 5000},
    'L2': {'seed_start': 6000},
    'L3': {'seed_start': 4000},
    'L4': {'seed_start': 1000},
    'L5': {'seed_start': 2000},
}


def main():
    parser = argparse.ArgumentParser(description='Generate EMRB benchmark problems')
    parser.add_argument('--level', required=True, choices=['L1', 'L2', 'L3', 'L4', 'L5'])
    parser.add_argument('--num', type=int, default=40)
    parser.add_argument('--seed-start', type=int, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    level = args.level
    output_dir = args.output or f'data/{level}'
    seed_start = args.seed_start or LEVEL_DEFAULTS[level]['seed_start']

    module = __import__(f'generation.generate_{level.lower()}_batch', fromlist=['generate_batch'])

    if level == 'L4':
        module.generate_batch(num_problems=args.num, seed_start=seed_start, output_dir=output_dir)
    else:
        module.generate_batch(num=args.num, seed_start=seed_start, output_dir=output_dir)


if __name__ == '__main__':
    main()
