import numpy

from gaussian_process import GaussianProcess, GaussianCovariance, zero_mean_function, CInfinityChebyshevCovariance
from deadhead_simulator import DeadheadSimulator, DEFAULT_DEADHEAD_TIMES
from sequential_optimization import run_bayesopt


def gaussian_process_test():
  for CovarianceClass in (GaussianCovariance, CInfinityChebyshevCovariance):
    covariance = CovarianceClass(.5)
    true_probabilities = numpy.array([0.26,  0.14,  0.16,  0.38,  0.44,  0.38,  0.26,  0.36])
    noise_variance = numpy.full_like(true_probabilities, 1e-2)
    gaussian_process = GaussianProcess(
      DEFAULT_DEADHEAD_TIMES,
      true_probabilities,
      covariance,
      zero_mean_function,
      .543,
      noise_variance,
    )
    posterior_draws = gaussian_process.posterior_draws(12)
    assert posterior_draws.shape == (12, gaussian_process.num_sampled)

    for _ in range(3):
      posterior_draws = gaussian_process.posterior_draws(1234)
      mean = numpy.mean(posterior_draws, axis=0)
      std = numpy.std(posterior_draws, axis=0)
      lower_bound = mean - 2 * std
      upper_bound = mean + 2 * std
      if numpy.all(numpy.logical_and(lower_bound < true_probabilities, true_probabilities < upper_bound)):
        break
    else:
      print(lower_bound)
      print(true_probabilities)
      print(upper_bound)
      raise AssertionError('This should have been okay by now ...')


def deadhead_simulator_test():
  deadhead_simulator = DeadheadSimulator()
  test_deadhead_distribution = numpy.array([
    0.2000001,  0.2001925,  0.21283038,  0.30451319,  0.39959046, 0.33183043,  0.23870112,  0.20599289,
  ])
  deadhead_simulator.generate_gamma_deadhead_distribution()
  assert numpy.allclose(test_deadhead_distribution, deadhead_simulator.deadhead_distribution)
  deadhead_simulator.construct_predictions(seed=123)
  deadhead_times, deadhead_results = list(zip(*deadhead_simulator.deadhead_call_predictions_by_deadhead_time.items()))
  assert set(deadhead_times) == set(deadhead_simulator.deadhead_times.tolist())
  assert all(len(v) == deadhead_simulator.max_calls for v in deadhead_results)

  for _ in range(3):
    deadhead_simulator = DeadheadSimulator(max_calls=10000)
    deadhead_simulator.generate_gamma_deadhead_distribution()
    successes = numpy.zeros_like(deadhead_simulator.deadhead_times)
    calls = numpy.zeros_like(deadhead_simulator.deadhead_times)
    deadhead_simulator.construct_predictions()
    for call in range(deadhead_simulator.max_calls):
      time_index = call % deadhead_simulator.num_times
      result = deadhead_simulator.simulate_call(deadhead_simulator.deadhead_times[time_index])
      successes[time_index] += result
      calls[time_index] += 1
    mean = successes / calls
    std = numpy.sqrt(mean * (1 - mean) / calls)
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std
    true_probabilities = deadhead_simulator.deadhead_distribution
    if numpy.all(numpy.logical_and(lower_bound < true_probabilities, true_probabilities < upper_bound)):
      break
  else:
    print(lower_bound)
    print(true_probabilities)
    print(upper_bound)
    raise AssertionError('This should have been okay by now ...')


def basic_bayesopt_test():
  deadhead_simulator = DeadheadSimulator([4, 6], max_calls=20)
  deadhead_simulator.deadhead_distribution = numpy.array([1.0, 0.0])
  deadhead_simulator.construct_predictions()
  gp = run_bayesopt(deadhead_simulator, verbose=True)
  print(f'Hopefully, this has first value much higher than second value, {gp.y}')


def main():
  gaussian_process_test()
  deadhead_simulator_test()
  basic_bayesopt_test()


if __name__ == '__main__':
  main()
  print('Success!')
