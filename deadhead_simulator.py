import numpy
from scipy.special import gamma as gamma_function

DEFAULT_MAX_CALLS = 50
DEFAULT_DEADHEAD_TIMES = numpy.arange(3, 11)
DEFAULT_DEADHEAD_MEAN = 7.2
DEFAULT_DEADHEAD_VARIANCE = .9876


class CallsExhaustedError(ValueError):
  pass


class SimulatorNotPrepared(ValueError):
  pass


class TimeNotRecognized(KeyError):
  pass


class DeadheadSimulator(object):
  def __init__(self, deadhead_times=DEFAULT_DEADHEAD_TIMES, max_calls=DEFAULT_MAX_CALLS):
    self.deadhead_times = numpy.array(deadhead_times)
    self.max_calls = max_calls

    self.deadhead_distribution = None
    self.deadhead_call_predictions = None
    self.deadhead_call_predictions_by_deadhead_time = None
    self.deadhead_call_results = None

    self.num_calls_made = 0
    self.deadhead_times_requested = []
    self.deadhead_time_requests_responses = []
    self.deadhead_time_requests_counts = {deadhead_time: 0 for deadhead_time in self.deadhead_times}

  @property
  def num_times(self):
    return len(self.deadhead_times)

  def generate_gamma_deadhead_distribution(
    self,
    deadhead_mean=DEFAULT_DEADHEAD_MEAN,
    deadhead_variance=DEFAULT_DEADHEAD_VARIANCE,
  ):
    beta = deadhead_mean / deadhead_variance
    alpha = deadhead_mean * beta
    mode = (alpha - 1) / beta

    def yf_unnorm(x):  # Should log space for safety??
      return beta ** alpha / gamma_function(alpha) * x ** (alpha - 1) * numpy.exp(-beta * x)

    self.deadhead_distribution = yf_unnorm(self.deadhead_times) / yf_unnorm(mode) * .2 + .2

  def construct_predictions(self, seed=None):
    if self.deadhead_distribution is None:
      raise SimulatorNotPrepared('deadhead_distribution has not been set')
    if seed is not None:
      numpy.random.seed(seed)
    uniform_randoms = numpy.random.random((self.num_times, self.max_calls))
    self.deadhead_call_predictions = uniform_randoms < self.deadhead_distribution[:, None]
    self.deadhead_call_predictions_by_deadhead_time = {
      deadhead_time: self.deadhead_call_predictions[k] for k, deadhead_time in enumerate(self.deadhead_times)
    }

  def simulate_call(self, deadhead_time):
    if self.deadhead_call_predictions_by_deadhead_time is None:
      raise SimulatorNotPrepared('construct_predictions has not been called yet')
    if self.num_calls_made >= self.max_calls:
      return CallsExhaustedError('All calls already conducted')

    try:
      result = self.deadhead_call_predictions_by_deadhead_time[deadhead_time][self.num_calls_made]
    except KeyError as e:
      raise TimeNotRecognized from e
    self.deadhead_times_requested.append(deadhead_time)
    self.deadhead_time_requests_responses.append(result)
    self.deadhead_time_requests_counts[deadhead_time] += 1
    self.num_calls_made += 1
    return result

  def log_likelihood(self, predicted_distribution):
    y_dict = dict(zip(self.deadhead_times, predicted_distribution))
    log_likelihood = 0
    for bb, tt in zip(self.deadhead_time_requests_responses, self.deadhead_times_requested):
      log_likelihood += numpy.log(y_dict[tt] if bb else (1 - y_dict[tt]))
    return log_likelihood
