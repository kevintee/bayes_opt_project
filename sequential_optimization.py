import numpy
from scipy.stats import norm
from scipy.optimize import differential_evolution

from gaussian_process import GaussianProcess, GaussianCovariance, constant_mean_function

# process variance, length scale, nonzero mean (mean will get more complicated)
DEFAULT_HPARAMS = numpy.array([.876, .567, -.543])
DEFAULT_HPARAM_BOUNDS = [[.1, 1], [.01, 1], [-1, 1]]
DEFAULT_DIFFERENTIAL_EVOLUTION_MAXITER = 50
DEFAULT_UCB_PERCENTILE = 75  # The time with the highest value of this percentile gets the next selection

STRAT_THOMPSON_SAMPLING = 'thompson-sampling'
STRAT_UCB = 'ucb'
STRAT_KNOWLEDGE_GRADIENT = 'knowledge-gradient'
STRAT_ENTROPY_SEARCH = 'entropy-search'
STRAT_EI = 'ei'
STRAT_NEI = 'nei'

ALL_STRATS = [STRAT_UCB, STRAT_THOMPSON_SAMPLING, STRAT_EI, STRAT_ENTROPY_SEARCH, STRAT_KNOWLEDGE_GRADIENT, STRAT_NEI]
DEFAULT_STRAT = STRAT_THOMPSON_SAMPLING
DEFAULT_MC_DRAWS = 1000


def form_noise_variance_from_simulator(deadhead_simulator):
  return 1 / (1 + deadhead_simulator.deadhead_time_requests_counts)


def form_gaussian_process_from_hparams(hparams, deadhead_simulator, y=None, noise_variance=None):
  length_scale, process_variance, constant_mean = hparams
  y = numpy.zeros_like(deadhead_simulator.deadhead_times) if y is None else y
  covariance = GaussianCovariance(length_scale, process_variance, deadhead_simulator.deadhead_times)
  mean_function = lambda x: constant_mean_function(x, constant_mean)
  noise_variance = form_noise_variance_from_simulator(deadhead_simulator) if noise_variance is None else noise_variance
  return GaussianProcess(covariance.x, y, covariance, mean_function, noise_variance)


# Maybe should make this more general ... right now only works for constant mean, really
def fit_model_to_data(deadhead_simulator, de_maxiter=None):
  if not deadhead_simulator.num_calls_made:
    return form_gaussian_process_from_hparams(DEFAULT_HPARAMS, deadhead_simulator)

  de_maxiter = de_maxiter or DEFAULT_DIFFERENTIAL_EVOLUTION_MAXITER
  noise_variance = form_noise_variance_from_simulator(deadhead_simulator)

  # Come up with a noise_variance multiplier to attenuate this further?
  # Something better than the "+1" strat to deal with zeros?
  def func(joint_vector):
    y = joint_vector[:deadhead_simulator.num_times]
    hparams = joint_vector[deadhead_simulator.num_times:]
    gaussian_process = form_gaussian_process_from_hparams(
      hparams,
      deadhead_simulator,
      y=y,
      noise_variance=noise_variance,
    )
    predicted_distribution = norm(loc=0, scale=1).cdf(y)

    log_hyperprior = 0  # Eventually could do something cooler
    log_gaussian_process_likelihood = gaussian_process.compute_log_likelihood()
    log_binomial_likelihood = deadhead_simulator.log_likelihood(predicted_distribution)

    return -(log_hyperprior + log_gaussian_process_likelihood + log_binomial_likelihood)

  bounds = numpy.array([[-2, 1]] * deadhead_simulator.num_times + DEFAULT_HPARAM_BOUNDS)
  result = differential_evolution(func, bounds, maxiter=de_maxiter)
  y = result.x[:deadhead_simulator.num_times]
  hparams = result.x[deadhead_simulator.num_times:]
  return form_gaussian_process_from_hparams(hparams, deadhead_simulator, y)


def choose_next_call(gaussian_process, **kwargs):
  strat = kwargs.get('opt_strat') or DEFAULT_STRAT
  if strat == STRAT_THOMPSON_SAMPLING:
    acquisition_function_values = norm(loc=0, scale=1).cdf(gaussian_process.posterior_draws(1).T)[:, 0]
  elif strat == STRAT_UCB:
    ucb_percentile = kwargs.get('ucb_percentile') or DEFAULT_UCB_PERCENTILE
    opt_mc_draws = kwargs.get('opt_mc_draws') or DEFAULT_MC_DRAWS
    z_draws = norm(loc=0, scale=1).cdf(gaussian_process.posterior_draws(opt_mc_draws).T)
    acquisition_function_values = numpy.percentile(z_draws, ucb_percentile, axis=1)
  else:
    raise ValueError('Unrecognized optimization strategy')
  return gaussian_process.x[numpy.argmax(acquisition_function_values)]


def run_bayesopt(deadhead_simulator, verbose=True, de_maxiter=None, **kwargs):
  for call in range(deadhead_simulator.max_calls):
    gaussian_process = fit_model_to_data(deadhead_simulator, de_maxiter=de_maxiter)
    next_deadhead_time = choose_next_call(gaussian_process, **kwargs)
    deadhead_simulator.simulate_call(next_deadhead_time)
    if verbose:
      print(f'Iteration {call}, tried {next_deadhead_time}')
  return fit_model_to_data(deadhead_simulator, de_maxiter=de_maxiter)