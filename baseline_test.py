import argparse
import numpy
import time

from baselines.base import ALL_BASELINES, run_all_baseline_trials_and_print_summary
from standard_test import form_deadhead_simulator_from_args


# Some way to structure randomness in the tests?
# Not sure what a good way to build different times into the tests.
# Would like to allow a mechanism for different distributions to be considered.
def initialize():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num-tests', type=int, required=True)
  parser.add_argument('--max-calls', type=int, required=True)
  parser.add_argument('--gamma-mean', type=float, required=False)
  parser.add_argument('--gamma-variance', type=float, required=False)
  parser.add_argument('--test-case', type=int, required=False, default=1)
  parser.add_argument('--baseline', type=str, required=True, choices=ALL_BASELINES)
  args = parser.parse_args()

  return args


# Init cycles could be better organized
def main():
  args = initialize()
  deadhead_simulator = form_deadhead_simulator_from_args(args)

  print(f'  Mean  |   SD   | Lower 25 | Upper 25')
  start = time.time()
  gaps = run_all_baseline_trials_and_print_summary(deadhead_simulator, args.baseline, args.num_tests)
  now = time.time()
  mean, sd, l25, u25 = numpy.mean(gaps), numpy.std(gaps), numpy.percentile(gaps, 25), numpy.percentile(gaps, 75)
  print(f' {mean:.4f}   {sd:.4f}    {l25:.4f}     {u25:.4f}')
  print(f'Tests complete, elapsed time: {now - start:.2f} seconds')


if __name__ == '__main__':
  main()

# Can be run with
#   python baseline_test.py --num-tests 200 --max-calls 50 --baseline osub --test-case 2
