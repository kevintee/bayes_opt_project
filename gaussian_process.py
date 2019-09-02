import numpy
from scipy.linalg import cho_solve, cho_factor, solve_triangular, cholesky


def zero_mean_function(x):
  return numpy.zeros_like(x)


def constant_mean_function(x, mean_value):
  return numpy.full_like(x, mean_value)


# Could definitely clean this up
class GaussianCovariance(object):
  def __init__(self, length_scale, process_variance, x):
    self.length_scale = length_scale
    self.process_variance = process_variance
    self.x = x
    self.symmetric_distance_matrix_squared = (x[:, None] - x[None, :]) ** 2

  def evaluate_covariance_given_distance_matrix_squared(self, distance_matrix_squared=None):
    dm2 = self.symmetric_distance_matrix_squared if distance_matrix_squared is None else distance_matrix_squared
    return self.process_variance * numpy.exp(-dm2 / self.length_scale ** 2)

  def evaluate_kernel_matrix(self, x_eval):
    distance_matrix_squared = (x_eval[:, None] - self.x[None, :]) ** 2
    return self.evaluate_covariance_given_distance_matrix_squared(distance_matrix_squared)


# This is only going to work in one dimension -- will consider other stuff eventually, maybe
class GaussianProcess(object):
  def __init__(self, x, y, covariance, mean_function, noise_variance):
    self.x = x
    self.y = y
    self.covariance = covariance
    self.mean_function = mean_function
    self.noise_variance = noise_variance

    self.prior_mean_values = None
    self.kernel_matrix = None
    self.kernel_matrix_plus_noise = None
    self.cho_kernel = None
    self.coef = None

    self._setup()

  def _setup(self):
    self.prior_mean_values = self.mean_function(self.x)
    self.kernel_matrix = self.covariance.evaluate_covariance_given_distance_matrix_squared()
    self.kernel_matrix_plus_noise = self.kernel_matrix + numpy.diag(self.noise_variance)
    self.cho_kernel = cho_factor(self.kernel_matrix_plus_noise, lower=True, overwrite_a=False)
    self.y_minus_mean = self.y - self.prior_mean_values
    self.coef = cho_solve(self.cho_kernel, self.y_minus_mean)

  @property
  def num_sampled(self):
    return len(self.y)

  def posterior_draws(self, num_draws):
    posterior_mean = self.prior_mean_values + numpy.dot(self.kernel_matrix, self.coef)
    temp = solve_triangular(self.cho_kernel[0], self.kernel_matrix.T, lower=self.cho_kernel[1], overwrite_b=False)
    posterior_cholesky = cholesky(self.kernel_matrix - numpy.dot(temp.T, temp), lower=True)
    normals = numpy.random.normal(size=(num_draws, self.num_sampled))
    return posterior_mean[None, :] + numpy.dot(normals, posterior_cholesky)

  def compute_log_likelihood(self):
    return (
      -numpy.dot(self.y_minus_mean, cho_solve(self.cho_kernel, self.y_minus_mean)) +
      -2 * numpy.sum(numpy.log(self.cho_kernel[0].diagonal()))
    )
