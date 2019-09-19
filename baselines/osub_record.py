import numpy


OSUB_NAME = 'osub'


# Adapted from Junpei's C++ code ... we should get the names synced up with the rest of the code base
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


def run_one_osub_baseline_trial(deadhead_simulator):
  o = OSUBPolicy(deadhead_simulator.num_times)
  for call_count in range(deadhead_simulator.max_calls):
    time_index = o.select_next_arm()
    result = deadhead_simulator.simulate_call(float(deadhead_simulator.deadhead_times[time_index]))
    o.update_state(time_index, result)
