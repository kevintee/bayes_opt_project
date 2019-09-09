#  This is what was run as a baseline -- could convert it into our format eventually

from scipy.special import gamma as gamma_function
import numpy


class OSUBPolicy(object):
  def __init__(self, K):
    self.gamma = 2
    self.delta = 1e-8
    self.epsilon = 1e-12
    self.K = K
    self.Ni = numpy.zeros(K)  # Number of queries per lever
    self.Gi = numpy.zeros(K)  # Number of successes per lever

  @staticmethod
  def kl(p, q):  # calculate kl-divergence
    return p * numpy.log(p / q) + (1 - p) * numpy.log((1 - p) / (1 - q))

  @staticmethod
  def dkl(p, q):
    return (q - p) / (q * (1.0 - q))

  def get_klucb_upper(self, k, n):
    logndn = numpy.log(n) / self.Ni[k]
    p = max(self.Gi[k] / self.Ni[k], self.delta)
    q = p + self.delta
    if p >= 1:
      return 1

    converged = False
    t = 0
    while t < 20 and not converged:
      f = logndn - self.kl(p, q)
      df = - self.dkl(p, q)
      if f ** 2 < self.epsilon:
        converged = True
        break
      q = min(1 - self.delta, max(q - f / df, p + self.delta))
      t += 1

    if not converged:
      raise AssertionError(f'Newton iteration in KL-UCB algorithm did not converge!! p={p}, logndn={logndn}')
    return q

  def select_next_arm(self):
    t = sum(self.Ni)
    indices = numpy.empty(self.K)
    for k in range(self.K):
      if self.Ni[k] == 0:
        return k
      # KL-UCB index
      indices[k] = self.get_klucb_upper(k, t)

    Lt = numpy.argmax(self.Ni)
    if self.Ni[Lt] < t / (1.0 + self.gamma):
      return Lt

    cur = Lt
    if Lt > 0 and indices[cur] < indices[Lt - 1]:
      cur = Lt - 1
    if Lt + 1 < self.K and indices[cur] < indices[Lt + 1]:
      cur = Lt + 1
    return cur

  def update_state(self, k, r):
    self.Ni[k] += 1
    self.Gi[k] += float(r)


# The actual running component is below -- should be boxed up but I'm just saving it here for now

deadhead_means = (7.2, 5.4, 8)
deadhead_variances = (0.9876, .2, 5)
deadhead_times = numpy.arange(3, 11)
hist_buckets = numpy.arange(3, 12) - .5

max_calls = 50
num_tries = 200

for deadhead_mean, deadhead_variance in zip(deadhead_means, deadhead_variances):
  wins = []
  gaps = []

  beta = deadhead_mean / deadhead_variance
  alpha = deadhead_mean * beta
  mode = (alpha - 1) / beta


  def yf_unnorm(x):  # Should log space for safety??
    return beta ** alpha / gamma_function(alpha) * x ** (alpha - 1) * numpy.exp(-beta * x)

  deadhead_distribution = yf_unnorm(deadhead_times) / yf_unnorm(mode) * .2 + .2

  for now in range(num_tries):
    calls = numpy.random.random(max_calls)
    o = OSUBPolicy(len(deadhead_times))
    for call_count in range(max_calls):
      time_index = o.select_next_arm()
      result = calls[call_count] < deadhead_distribution[time_index]
      o.update_state(time_index, result)

    results = o.Gi / (o.Ni + 1e-10)
    winning_indices = numpy.where(results == max(results))[0]
    winning_index = numpy.random.choice(winning_indices)
    wins.append(deadhead_times[winning_index])
    gaps.append(max(deadhead_distribution) - deadhead_distribution[winning_index])

  print(numpy.mean(gaps), numpy.std(gaps), numpy.percentile(gaps, 25), numpy.percentile(gaps, 75))
