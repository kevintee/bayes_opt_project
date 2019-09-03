import argparse
import json
import numpy
import os
import time

from deadhead_simulator import DeadheadSimulator, DEFAULT_DEADHEAD_MEAN, DEFAULT_DEADHEAD_VARIANCE
from sequential_optimization import run_bayesopt, ALL_STRATS


# Some way to structure randomness in the tests?
# Not sure what a good way to build different times into the tests.
# Would like to allow a mechanism for different distributions to be considered.
def initialize():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output-file', type=str, required=False)
  parser.add_argument('--num-tests', type=int, required=True)
  parser.add_argument('--max-calls', type=int, required=True)
  parser.add_argument('--verbosity', type=int, required=False, default=2)
  parser.add_argument('--gamma-mean', type=float, required=False, default=DEFAULT_DEADHEAD_MEAN)
  parser.add_argument('--gamma-variance', type=float, required=False, default=DEFAULT_DEADHEAD_VARIANCE)
  parser.add_argument('--de-maxiter', type=int, required=False)
  parser.add_argument('--opt-strat', type=str, required=False, choices=ALL_STRATS)
  parser.add_argument('--opt-mc-draws', type=int, required=False)
  parser.add_argument('--ucb-percentile', type=float, required=False)
  args = parser.parse_args()

  return args


def write_results(output_file, deadhead_simulator, create_file=False, answer=None, gaussian_process=None):
  if create_file:
    if os.path.isfile(output_file):
      raise AssertionError(f'File {output_file} already exists!')
    info = {
      'deadhead_times': deadhead_simulator.deadhead_times.tolist(),
      'deadhead_distribution': deadhead_simulator.deadhead_distribution.tolist(),
      'results': [],
    }
    with open(output_file, 'w') as f:
      json.dump(info, f)
    return

  assert deadhead_simulator.num_calls_made == deadhead_simulator.max_calls

  with open(output_file, 'r') as f:
    info = json.load(f)

  new_results = {
    'deadhead_times_requested': deadhead_simulator.deadhead_times_requested,
    'deadhead_time_requests_responses': deadhead_simulator.deadhead_time_requests_responses,
  }
  if answer is not None:
    new_results['answer'] = float(answer)  # Casting for json
  if gaussian_process is not None:
    new_results['gaussian_process'] = {
      'y': gaussian_process.y.tolist(),
      'process_variance': gaussian_process.process_variance,
      'length_scale': gaussian_process.length_scale,
    }

  info['results'].append(new_results)

  with open(output_file, 'w') as f:
    json.dump(info, f)


def form_next_call_kwargs(args):
  return {
    'opt_strat': args.opt_strat,
    'opt_mc_draws': args.opt_mc_draws,
    'ucb_percentile': args.ucb_percentile,
  }


def main():
  args = initialize()
  output_file = args.output_file or f'{time.time()}_{args.max_calls}.json'

  deadhead_simulator = DeadheadSimulator(max_calls=args.max_calls)
  deadhead_simulator.generate_gamma_deadhead_distribution(
    deadhead_mean=args.gamma_mean,
    deadhead_variance=args.gamma_variance,
  )
  write_results(output_file, deadhead_simulator, create_file=True)
  start = time.time()
  for k in range(args.num_tests):
    if args.verbosity:
      print(f'\tTrial {k} starting')
    deadhead_simulator.construct_predictions()
    gaussian_process = run_bayesopt(
      deadhead_simulator,
      verbose=args.verbosity > 1,
      de_maxiter=args.de_maxiter,
      **form_next_call_kwargs(args),
    )
    answer = deadhead_simulator.deadhead_times[numpy.argmax(gaussian_process.y)]
    write_results(output_file, deadhead_simulator, answer=answer, gaussian_process=gaussian_process)
    if args.verbosity:
      now = time.time()
      print(f'\tTrial {k} complete, elapsed time: {now - start:.2f} seconds')


if __name__ == '__main__':
  main()

# Can be run with
#   python standard_test.py --num-tests 4 --max-calls 50 --de-maxiter 25 --output-file test.json
# Hopefully the file test.json is easily interpretable