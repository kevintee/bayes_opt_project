import numpy
from scipy.stats import norm
from scipy.optimize import differential_evolution

from gaussian_process import (
  GaussianProcess, GaussianCovariance, ConstantMean, CInfinityChebyshevCovariance,
  DEFAULT_LENGTH_SCALE, DEFAULT_PROCESS_VARIANCE, DEFAULT_CONSTANT_MEAN,
  DEFAULT_LENGTH_SCALE_HPARAM_BOUNDS, DEFAULT_PROCESS_VARIANCE_HPARAM_BOUNDS, DEFAULT_CONSTANT_MEAN_HPARAM_BOUNDS,
)

DEFAULT_HPARAMS = (DEFAULT_LENGTH_SCALE, DEFAULT_PROCESS_VARIANCE, DEFAULT_CONSTANT_MEAN)
DEFAULT_HPARAM_BOUNDS = [
  DEFAULT_LENGTH_SCALE_HPARAM_BOUNDS,
  DEFAULT_PROCESS_VARIANCE_HPARAM_BOUNDS,
  DEFAULT_CONSTANT_MEAN_HPARAM_BOUNDS,
]

DEFAULT_DIFFERENTIAL_EVOLUTION_MAXITER = 16
DEFAULT_UCB_PERCENTILE = 75  # The time with the highest value of this percentile gets the next selection
DEFAULT_AEI_PERCENTILE = 55  # The time with the highest value of this percentile is assigned "best seen" status for EI

STRAT_THOMPSON_SAMPLING = 'thompson-sampling'
STRAT_UCB = 'ucb'
STRAT_KNOWLEDGE_GRADIENT = 'knowledge-gradient'
STRAT_ENTROPY_SEARCH = 'entropy-search'
STRAT_EI = 'ei'
STRAT_AEI = 'aei'

ALL_STRATS = [STRAT_UCB, STRAT_THOMPSON_SAMPLING, STRAT_EI, STRAT_ENTROPY_SEARCH, STRAT_KNOWLEDGE_GRADIENT, STRAT_AEI]
DEFAULT_STRAT = STRAT_THOMPSON_SAMPLING
DEFAULT_MC_DRAWS = 1000

ALL_COVARIANCES = {x.name: x for x in (GaussianCovariance, CInfinityChebyshevCovariance)}
DEFAULT_COVARIANCE = GaussianCovariance.name


def form_noise_variance_from_simulator(deadhead_simulator):
  return 1 / (1 + deadhead_simulator.deadhead_time_requests_counts)


def form_gaussian_process_from_hparams(hparams, deadhead_simulator, noise_variance, y=None, **kwargs):
  length_scale, process_variance, constant_mean = hparams
  x = deadhead_simulator.deadhead_times
  y = numpy.zeros_like(x) if y is None else y

  covariance_name = kwargs.get('gp_covariance') or DEFAULT_COVARIANCE
  covariance = ALL_COVARIANCES[covariance_name](length_scale)

  mean_function = ConstantMean(constant_mean)
  return GaussianProcess(x, y, covariance, mean_function, process_variance, noise_variance)


# Maybe should make this more general ... right now only works for constant mean, really
def fit_model_to_data(deadhead_simulator, **kwargs):
  noise_variance = form_noise_variance_from_simulator(deadhead_simulator)
  if not deadhead_simulator.num_calls_made:
    return form_gaussian_process_from_hparams(DEFAULT_HPARAMS, deadhead_simulator, noise_variance, **kwargs)

  de_maxiter = kwargs.get('de_maxiter') or DEFAULT_DIFFERENTIAL_EVOLUTION_MAXITER

  # Come up with a noise_variance multiplier to attenuate this further?
  # Something better than the "+1" strat to deal with zeros?
  def func(joint_vector):
    y = joint_vector[:deadhead_simulator.num_times]
    hparams = joint_vector[deadhead_simulator.num_times:]
    gaussian_process = form_gaussian_process_from_hparams(hparams, deadhead_simulator, noise_variance, y, **kwargs)
    predicted_distribution = norm(loc=0, scale=1).cdf(y)

    log_hyperprior = 0  # Eventually could do something cooler
    log_gaussian_process_likelihood = gaussian_process.compute_log_likelihood()
    log_binomial_likelihood = deadhead_simulator.log_likelihood(predicted_distribution)

    return -(log_hyperprior + log_gaussian_process_likelihood + log_binomial_likelihood)

  bounds = numpy.array([[-2, 1]] * deadhead_simulator.num_times + DEFAULT_HPARAM_BOUNDS)
  result = differential_evolution(func, bounds, maxiter=de_maxiter)
  y = result.x[:deadhead_simulator.num_times]
  hparams = result.x[deadhead_simulator.num_times:]
  return form_gaussian_process_from_hparams(hparams, deadhead_simulator, noise_variance, y, **kwargs)


def choose_next_call(gaussian_process, **kwargs):
  strat = kwargs.get('opt_strat') or DEFAULT_STRAT
  opt_mc_draws = kwargs.get('opt_mc_draws') or DEFAULT_MC_DRAWS
  if strat == STRAT_THOMPSON_SAMPLING:
    acquisition_function_values = norm(loc=0, scale=1).cdf(gaussian_process.posterior_draws(1).T)[:, 0]
  elif strat == STRAT_UCB:
    ucb_percentile = kwargs.get('ucb_percentile') or DEFAULT_UCB_PERCENTILE
    z_draws = norm(loc=0, scale=1).cdf(gaussian_process.posterior_draws(opt_mc_draws).T)
    acquisition_function_values = numpy.percentile(z_draws, ucb_percentile, axis=1)
  elif strat == STRAT_EI:
    z_draws = norm(loc=0, scale=1).cdf(gaussian_process.posterior_draws(opt_mc_draws).T)
    best_observed_value = numpy.max(gaussian_process.y)
    acquisition_function_values = numpy.mean(numpy.fmax(z_draws - best_observed_value, 0), axis=1)
  elif strat == STRAT_AEI:
    aei_percentile = kwargs.get('aei_percentile') or DEFAULT_AEI_PERCENTILE
    z_draws = norm(loc=0, scale=1).cdf(gaussian_process.posterior_draws(opt_mc_draws).T)
    best_noisy_observed_value = numpy.max(numpy.percentile(z_draws, aei_percentile, axis=1))
    z_draws = norm(loc=0, scale=1).cdf(gaussian_process.posterior_draws(opt_mc_draws).T)
    expected_improvement = numpy.mean(numpy.fmax(z_draws - best_noisy_observed_value, 0), axis=1)
    z_draws = norm(loc=0, scale=1).cdf(gaussian_process.posterior_draws(opt_mc_draws).T)
    posterior_variance = numpy.var(z_draws, axis=1)
    mean_noise_variance = numpy.mean(gaussian_process.noise_variance)
    penalty = 1 - numpy.sqrt(mean_noise_variance / (posterior_variance + mean_noise_variance))
    acquisition_function_values = expected_improvement * penalty
  else:
    raise ValueError('Unrecognized optimization strategy')
  return gaussian_process.x[numpy.argmax(acquisition_function_values)]


def run_bayesopt(deadhead_simulator, verbose=True, **kwargs):
  for call in range(deadhead_simulator.max_calls):
    gaussian_process = fit_model_to_data(deadhead_simulator, **kwargs)
    next_deadhead_time = choose_next_call(gaussian_process, **kwargs)
    deadhead_simulator.simulate_call(next_deadhead_time)
    draws = gaussian_process.posterior_draws(1000)
    if verbose:
      print(f'Iteration {call}, tried {next_deadhead_time}')
  return fit_model_to_data(deadhead_simulator, **kwargs)
