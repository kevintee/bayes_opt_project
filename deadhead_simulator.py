import numpy
from copy import deepcopy
from scipy.special import gamma as gamma_function

DEFAULT_MAX_CALLS = 50
DEFAULT_DEADHEAD_TIMES = numpy.arange(3, 11)
DEFAULT_DEADHEAD_MEAN, DEFAULT_DEADHEAD_VARIANCE = 7.2, 0.9876


class CallsExhaustedError(ValueError):
  pass


class SimulatorNotPrepared(ValueError):
  pass


class TimeNotRecognized(KeyError):
  pass


class DeadheadSimulator(object):
  def __init__(self, deadhead_times=DEFAULT_DEADHEAD_TIMES, max_calls=DEFAULT_MAX_CALLS):
    assert len(set(deadhead_times)) == len(deadhead_times), 'Non-unique deadhead times detected'
    self.deadhead_times = numpy.array(deadhead_times)
    self.max_calls = max_calls
    self.deadhead_time_indexes = dict(zip(self.deadhead_times, range(self.num_times)))

    self.deadhead_distribution = None
    self.deadhead_call_predictions = None
    self.deadhead_call_predictions_by_deadhead_time = None

    self.num_calls_made = None
    # A record of what happened in the order it happened
    self.deadhead_times_requested = None
    self.deadhead_time_requests_responses = None
    # An accumulation of the counts of what occurred, in arrays to match the deadhead times
    self.deadhead_time_requests_counts = None
    self.deadhead_time_successes_counts = None
    self._reset_record()

  def _reset_record(self):
    self.num_calls_made = 0
    self.deadhead_times_requested = []
    self.deadhead_time_requests_responses = []
    self.deadhead_time_requests_counts = numpy.zeros_like(self.deadhead_times)
    self.deadhead_time_successes_counts = numpy.zeros_like(self.deadhead_times)

  def highest_performing_times(self):
    results = self.deadhead_time_successes_counts / (self.deadhead_time_requests_counts + 1e-10)
    winning_indices = numpy.where(results == max(results))[0]
    return self.deadhead_times[winning_indices]

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
    self.num_calls_made = 0
    if seed is not None:
      numpy.random.seed(seed)
    uniform_randoms = numpy.random.random((self.num_times, self.max_calls))
    self.deadhead_call_predictions = uniform_randoms < self.deadhead_distribution[:, None]
    self.deadhead_call_predictions_by_deadhead_time = {
      deadhead_time: self.deadhead_call_predictions[k] for k, deadhead_time in enumerate(self.deadhead_times)
    }
    self._reset_record()

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
    self.deadhead_time_requests_responses.append(bool(result))  # Casting for json

    deadhead_time_index = self.deadhead_time_indexes[deadhead_time]
    self.deadhead_time_requests_counts[deadhead_time_index] += 1
    self.deadhead_time_successes_counts[deadhead_time_index] += int(result)
    self.num_calls_made += 1
    return result

  def simulate_forced_call_result(self, deadhead_time, result):
    try:
      self.deadhead_call_predictions_by_deadhead_time[deadhead_time][self.num_calls_made] = result
    except KeyError as e:
      raise TimeNotRecognized from e
    return self.simulate_call(deadhead_time)

  # Could be made more efficient by storing data as array rather than dict -- maybe worth considering
  def log_likelihood(self, predicted_distribution):
    return numpy.sum(
      self.deadhead_time_successes_counts * numpy.log(predicted_distribution) +
      self.deadhead_time_requests_counts * numpy.log(1 - predicted_distribution)
    )


def duplicate_deadhead_simulator(deadhead_simulator):
  assert isinstance(deadhead_simulator, DeadheadSimulator)
  new_ds = DeadheadSimulator(deadhead_simulator.deadhead_times, deadhead_simulator.max_calls)

  new_ds.deadhead_distribution = deepcopy(deadhead_simulator.deadhead_distribution)
  new_ds.deadhead_call_predictions = deepcopy(deadhead_simulator.deadhead_call_predictions)
  new_ds.deadhead_call_predictions_by_deadhead_time = deepcopy(deadhead_simulator.deadhead_call_predictions_by_deadhead_time)

  new_ds.num_calls_made = deepcopy(deadhead_simulator.num_calls_made)
  new_ds.deadhead_times_requested = deepcopy(deadhead_simulator.deadhead_times_requested)
  new_ds.deadhead_time_requests_responses = deepcopy(deadhead_simulator.deadhead_time_requests_responses)
  new_ds.deadhead_time_requests_counts = deepcopy(deadhead_simulator.deadhead_time_requests_counts)
  new_ds.deadhead_time_successes_counts = deepcopy(deadhead_simulator.deadhead_time_successes_counts)

  return new_ds
