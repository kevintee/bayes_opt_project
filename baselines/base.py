import numpy

from baselines.osub_record import OSUB_NAME, run_one_osub_baseline_trial
from baselines.lse_demo import LSE_NAME, run_one_lse_baseline_trial
from baselines.ucbe_baseline import UCBE_NAME, run_one_ucbe_baseline_trial
from deadhead_simulator import DeadheadSimulator


ALL_BASELINES = (OSUB_NAME, LSE_NAME, UCBE_NAME)


def run_one_baseline_trial(deadhead_simulator, baseline_name):
  assert isinstance(deadhead_simulator, DeadheadSimulator) and baseline_name in ALL_BASELINES

  if baseline_name == LSE_NAME:  # Limited for now
    assert numpy.array_equal(deadhead_simulator.deadhead_times, numpy.arange(3, 11))
    run_one_lse_baseline_trial(deadhead_simulator)
  elif baseline_name == OSUB_NAME:
    run_one_osub_baseline_trial(deadhead_simulator)
  elif baseline_name == UCBE_NAME:
    run_one_ucbe_baseline_trial(deadhead_simulator)


def run_all_baseline_trials_and_print_summary(deadhead_simulator, baseline_name, num_trials):
  assert isinstance(deadhead_simulator, DeadheadSimulator) and baseline_name in ALL_BASELINES
  max_prob = max(deadhead_simulator.deadhead_distribution)

  if baseline_name == LSE_NAME:  # Limited for now
    assert numpy.array_equal(deadhead_simulator.deadhead_times, numpy.arange(3, 11))
    runner = run_one_lse_baseline_trial
  elif baseline_name == OSUB_NAME:
    runner = run_one_osub_baseline_trial
  elif baseline_name == UCBE_NAME:
    runner = run_one_ucbe_baseline_trial

  gaps = []
  for _ in range(num_trials):
    deadhead_simulator.construct_predictions()
    runner(deadhead_simulator)
    answer = numpy.random.choice(deadhead_simulator.highest_performing_times())
    prob_of_answer = deadhead_simulator.deadhead_distribution[deadhead_simulator.deadhead_time_indexes[answer]]
    gaps.append(max_prob - prob_of_answer)

  return gaps
