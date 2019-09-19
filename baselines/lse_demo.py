import numpy
from deadhead_simulator import CallsExhaustedError

LSE_NAME = 'lse'

# Shouldn't actually use this, just copying it here for a moment

# This is required at the moment -- can eventually do something more flexible
REQUIRED_MAX_CALLS = 50
REQUIRED_DEADHEAD_TIMES = numpy.arange(3, 11)

def run_one_lse_baseline_trial(deadhead_simulator):
  assert deadhead_simulator.max_calls == REQUIRED_MAX_CALLS
  assert numpy.array_equal(deadhead_simulator.deadhead_times, REQUIRED_DEADHEAD_TIMES)

  # Determined by the fixed 50 calls and 8 arms ... could be made more systemic
  trials_per_arm = 8

  times_under_consideration = [3, 6, 8, 10]
  for this_time in times_under_consideration:
    for _ in range(trials_per_arm):
      deadhead_simulator.simulate_call(this_time)
  best_time_thus_far = numpy.random.choice(deadhead_simulator.highest_performing_times())

  if best_time_thus_far < 7:
    for _ in range(trials_per_arm):
      deadhead_simulator.simulate_call(5)
    best_time_thus_far = numpy.random.choice(deadhead_simulator.highest_performing_times())
    if best_time_thus_far < 6:
      for _ in range(trials_per_arm):
        deadhead_simulator.simulate_call(4)
      final_times_for_polish = [3, 4, 5, 6]
    else:
      for _ in range(trials_per_arm):
        deadhead_simulator.simulate_call(7)
      final_times_for_polish = [6, 7, 8]
  else:
    for _ in range(trials_per_arm):
      deadhead_simulator.simulate_call(7)
      deadhead_simulator.simulate_call(9)
    final_times_for_polish = [6, 7, 8, 9, 10]

  while True:
    try:
      deadhead_simulator.simulate_call(numpy.random.choice(final_times_for_polish))
    except CallsExhaustedError:
      break

  assert deadhead_simulator.num_calls_made == deadhead_simulator.max_calls
