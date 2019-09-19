import numpy
from deadhead_simulator import CallsExhaustedError

UCBE_NAME = 'ucbe'

def run_one_ucbe_baseline_trial(deadhead_simulator):
  # This exploration parameter needs to exist, but guidance on choosing it is limited
  # This is a quantity that was somewhat suggested in the UCB-E paper
  H1 = sum(max(deadhead_simulator.deadhead_distribution) - deadhead_simulator.deadhead_distribution)
  exploration = .5 * 25 / 36 * (deadhead_simulator.max_calls - deadhead_simulator.num_times) / H1

  for this_time in deadhead_simulator.deadhead_times:
    deadhead_simulator.simulate_call(this_time)

  while True:
    successes = deadhead_simulator.deadhead_time_successes_counts
    tries = deadhead_simulator.deadhead_time_requests_counts
    ucb_vals = successes / tries + numpy.sqrt(exploration / tries)
    next_index_to_test = numpy.random.choice(numpy.where(max(ucb_vals) == ucb_vals)[0])
    try:
      deadhead_simulator.simulate_call(deadhead_simulator.deadhead_times[next_index_to_test])
    except CallsExhaustedError:
      break

  assert deadhead_simulator.num_calls_made == deadhead_simulator.max_calls
