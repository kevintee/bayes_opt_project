import numpy
from scipy.stats import norm
from scipy.optimize import differential_evolution, minimize

from gaussian_process import (
  GaussianProcess, GaussianCovariance, CInfinityChebyshevCovariance,
  ZeroMean, ConstantMean, BellCurveMean, FixedBellCurveMean,
  DEFAULT_PROCESS_VARIANCE, DEFAULT_PROCESS_VARIANCE_HPARAM_BOUNDS,
)
from deadhead_simulator import duplicate_deadhead_simulator


DEFAULT_DIFFERENTIAL_EVOLUTION_MAXITER = 16
DEFAULT_UCB_PERCENTILE = 75  # The time with the highest value of this percentile gets the next selection
DEFAULT_AEI_PERCENTILE = 55  # The time with the highest value of this percentile is assigned "best seen" status for EI
DEFAULT_KG_PERCENTILE = 65  # Instead of just considering the highest expected value, free it up some

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

ALL_MEAN_FUNCTIONS = {x.name: x for x in (ZeroMean, ConstantMean, BellCurveMean, FixedBellCurveMean)}
DEFAULT_MEAN_FUNCTION = ConstantMean.name


# Come up with a noise_variance multiplier to attenuate this further?
# Something better than the "+1" strat to deal with zeros?
def form_noise_variance_from_simulator(deadhead_simulator):
  return 1 / (1 + deadhead_simulator.deadhead_time_requests_counts)


def form_gaussian_process(
  hparams,
  deadhead_simulator,
  y,
  noise_variance,
  covariance_class,
  mean_function_class,
):
  length_scale, process_variance, *mean_function_args = hparams

  x = deadhead_simulator.deadhead_times
  covariance = covariance_class(length_scale)
  mean_function = mean_function_class(*mean_function_args)

  return GaussianProcess(x, y, covariance, mean_function, process_variance, noise_variance)


def fit_model_to_data(deadhead_simulator, current_gaussian_process=None, **kwargs):
  covariance_name = kwargs.get('gp_covariance') or DEFAULT_COVARIANCE
  covariance_class = ALL_COVARIANCES[covariance_name]
  mean_function_name = kwargs.get('mean_function') or DEFAULT_MEAN_FUNCTION
  mean_function_class = ALL_MEAN_FUNCTIONS[mean_function_name]
  noise_variance = form_noise_variance_from_simulator(deadhead_simulator)

  def form_gaussian_process_from_hparams(y, hparams):
    return form_gaussian_process(
      hparams,
      deadhead_simulator,
      y,
      noise_variance,
      covariance_class,
      mean_function_class,
    )

  if not deadhead_simulator.num_calls_made:
    default_hparams = [None, DEFAULT_PROCESS_VARIANCE]
    return form_gaussian_process_from_hparams(numpy.zeros(deadhead_simulator.num_times), default_hparams)

  def func(joint_vector):
    y = joint_vector[:deadhead_simulator.num_times]
    hparams = joint_vector[deadhead_simulator.num_times:]
    gaussian_process = form_gaussian_process_from_hparams(y, hparams)
    predicted_distribution = norm(loc=0, scale=1).cdf(y)

    log_hyperprior = 0  # Eventually could do something cooler
    log_gaussian_process_likelihood = gaussian_process.compute_log_likelihood()
    log_bernoulli_likelihood = deadhead_simulator.log_likelihood(predicted_distribution)

    return -(log_hyperprior + log_gaussian_process_likelihood + log_bernoulli_likelihood)

  bounds = (
    [[-2, 1]] * deadhead_simulator.num_times +  # y bounds
    [covariance_class.DEFAULT_LENGTH_SCALE_BOUNDS] +
    [DEFAULT_PROCESS_VARIANCE_HPARAM_BOUNDS] +
    mean_function_class.DEFAULT_HPARAM_BOUNDS
  )

  if current_gaussian_process is not None:  # We are hallucinating the fake data for knowledge gradient
    initial_guess = current_gaussian_process.recover_tunable_values_for_initial_guess()
    result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds)
  else:
    de_maxiter = kwargs.get('de_maxiter') or DEFAULT_DIFFERENTIAL_EVOLUTION_MAXITER
    result = differential_evolution(func, bounds, maxiter=de_maxiter)
  y = result.x[:deadhead_simulator.num_times]
  hparams = result.x[deadhead_simulator.num_times:]

  return form_gaussian_process_from_hparams(y, hparams)


def choose_next_call(deadhead_simulator, gaussian_process, **kwargs):
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
  elif strat == STRAT_KNOWLEDGE_GRADIENT:
    kg_percentile = kwargs.get('kg_percentile') or DEFAULT_KG_PERCENTILE
    probabilities = norm(loc=0, scale=1).cdf(gaussian_process.y)
    acquisition_function_values = numpy.zeros_like(gaussian_process.x)
    for k, (test_time, expected_probability) in enumerate(zip(deadhead_simulator.deadhead_times, probabilities)):
      for projected_result, probability in zip((True, False), (expected_probability, 1 - expected_probability)):
        projected_ds = duplicate_deadhead_simulator(deadhead_simulator)
        projected_ds.simulate_forced_call_result(test_time, projected_result)
        projected_gp = fit_model_to_data(projected_ds, current_gaussian_process=gaussian_process, **kwargs)
        z_draws = norm(loc=0, scale=1).cdf(projected_gp.posterior_draws(opt_mc_draws).T)
        acquisition_function_values[k] += max(numpy.percentile(z_draws, kg_percentile, axis=1)) * probability
  else:
    raise ValueError('Unrecognized optimization strategy')

  return gaussian_process.x[numpy.argmax(acquisition_function_values)]


def run_bayesopt(deadhead_simulator, verbose=True, **kwargs):
  for call in range(deadhead_simulator.max_calls):
    gaussian_process = fit_model_to_data(deadhead_simulator, **kwargs)
    next_deadhead_time = choose_next_call(deadhead_simulator, gaussian_process, **kwargs)
    deadhead_simulator.simulate_call(next_deadhead_time)
    fit_model_to_data(deadhead_simulator, current_gaussian_process=gaussian_process, **kwargs)
    if verbose:
      print(f'Iteration {call}, tried {next_deadhead_time}')
  return fit_model_to_data(deadhead_simulator, **kwargs)
