import numpy
from scipy.stats import norm
from scipy.optimize import differential_evolution

from gaussian_process import GaussianProcess, GaussianCovariance, constant_mean_function

# process variance, length scale, nonzero mean (mean will get more complicated)
DEFAULT_HPARAMS = numpy.array([.876, .567, -.543])
DEFAULT_HPARAM_BOUNDS = [[.1, 1], [.01, 1], [-1, 1]]


def form_noise_variance_from_simulator(deadhead_simulator):
  noise_variance = numpy.ones_like(deadhead_simulator.deadhead_times, dtype=float)
  for k, deadhead_time in enumerate(deadhead_simulator.deadhead_times):
    noise_variance[k] = 1 / (1 + deadhead_simulator.deadhead_time_requests_counts[deadhead_time])
  return noise_variance


def form_gaussian_process_from_hparams(hparams, deadhead_simulator, y=None):
  length_scale, process_variance, constant_mean = hparams
  y = numpy.zeros_like(deadhead_simulator.deadhead_times) if y is None else y
  covariance = GaussianCovariance(length_scale, process_variance, deadhead_simulator.deadhead_times)
  mean_function = lambda x: constant_mean_function(x, constant_mean)
  noise_variance = form_noise_variance_from_simulator(deadhead_simulator)
  return GaussianProcess(covariance.x, y, covariance, mean_function, noise_variance)


# Maybe should make this more general ... right now only works for constant mean, really
def fit_model_to_data(deadhead_simulator):
  if not deadhead_simulator.num_calls_made:
    return form_gaussian_process_from_hparams(DEFAULT_HPARAMS, deadhead_simulator)

  # Come up with a noise_variance multiplier to attenuate this further?
  # Something better than the "+1" strat to deal with zeros?
  def func(joint_vector):
    y = joint_vector[:deadhead_simulator.num_times]
    hparams = joint_vector[deadhead_simulator.num_times:]
    gaussian_process = form_gaussian_process_from_hparams(hparams, deadhead_simulator, y)
    predicted_distribution = norm(loc=0, scale=1).cdf(y)

    log_hyperprior = 0  # Eventually could do something cooler
    log_gaussian_process_likelihood = gaussian_process.compute_log_likelihood()
    log_binomial_likelihood = deadhead_simulator.log_likelihood(predicted_distribution)

    return -(log_hyperprior + log_gaussian_process_likelihood + log_binomial_likelihood)

  bounds = numpy.array([[-2, 1]] * deadhead_simulator.num_times + DEFAULT_HPARAM_BOUNDS)
  result = differential_evolution(func, bounds, maxiter=100)
  y = result.x[:deadhead_simulator.num_times]
  hparams = result.x[deadhead_simulator.num_times:]
  return form_gaussian_process_from_hparams(hparams, deadhead_simulator, y)


# Just doing Thompson sampling for right now -- can do better later
def choose_next_call(gp):
  z_draws = norm(loc=0, scale=1).cdf(gp.posterior_draws(1).T)
  next_deadhead_time_index = numpy.argmax(z_draws, axis=0)[0]
  return gp.x[next_deadhead_time_index]


def run_bayesopt(deadhead_simulator, verbose=True):
  for call in range(deadhead_simulator.max_calls):
    gp = fit_model_to_data(deadhead_simulator)
    next_deadhead_time = choose_next_call(gp)
    deadhead_simulator.simulate_call(next_deadhead_time)
    if verbose:
      print(f'Iteration {call}, tried {next_deadhead_time}')
  return fit_model_to_data(deadhead_simulator)
