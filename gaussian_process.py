import numpy
from scipy.linalg import cho_solve, cho_factor, solve_triangular, cholesky
from scipy.stats import norm


DEFAULT_LENGTH_SCALE_HPARAM_BOUNDS = [.0987, .987]
DEFAULT_PROCESS_VARIANCE_HPARAM_BOUNDS = [.0432, .9012]
DEFAULT_CONSTANT_MEAN_HPARAM_BOUNDS = [-1.23, 1.098]
DEFAULT_BELL_CURVE_LOC_HPARAM_BOUNDS = [3.0, 10.0]
DEFAULT_BELL_CURVE_SCALE_HPARAM_BOUNDS = [.4321, 4.321]
DEFAULT_BELL_CURVE_MIN_HPARAM_BOUNDS = [-2.345, -.4321]
DEFAULT_BELL_CURVE_MAX_HPARAM_BOUNDS = [-.4567, .8765]

DEFAULT_LENGTH_SCALE = .567
DEFAULT_PROCESS_VARIANCE = .876

DEFAULT_CONSTANT_MEAN_ARGS = (-.543, )
DEFAULT_FIXED_BELL_CURVE_MEAN_ARGS = (6.5, 1.0)  # In reality, should have domain data to know the loc
DEFAULT_BELL_CURVE_MEAN_ARGS = (6.5, 1.0, -.85, -.25)


class ConstantMean(object):
  name = 'constant'
  DEFAULT_HPARAM_BOUNDS = [DEFAULT_CONSTANT_MEAN_HPARAM_BOUNDS]

  def __init__(self, *args, **kwargs):
    assert len(args) in (0, 1) and len(kwargs) == 0
    args = args or DEFAULT_CONSTANT_MEAN_ARGS
    self.mean_value = args[0]

  def __str__(self):
    return f'Constant({self.mean_value})'

  def __call__(self, *args, **kwargs):
    assert len(args) == 1 and len(kwargs) == 0
    (x, ) = args
    return numpy.full_like(x, self.mean_value)


class ZeroMean(ConstantMean):
  name = 'zero'
  DEFAULT_HPARAM_BOUNDS = []

  def __init__(self, *args, **kwargs):
    assert len(args) == 0 and len(kwargs) == 0
    super().__init__(0.0)


class BellCurveMean(object):
  name = 'bell-curve'
  DEFAULT_HPARAM_BOUNDS = [
    DEFAULT_BELL_CURVE_LOC_HPARAM_BOUNDS,
    DEFAULT_BELL_CURVE_SCALE_HPARAM_BOUNDS,
    DEFAULT_BELL_CURVE_MIN_HPARAM_BOUNDS,
    DEFAULT_BELL_CURVE_MAX_HPARAM_BOUNDS,
  ]

  def __init__(self, *args, **kwargs):
    assert len(args) in (0, 4) and len(kwargs) == 0
    args = args or DEFAULT_BELL_CURVE_MEAN_ARGS
    (loc, scale, min_val, max_val) = args
    self.loc = loc
    self.scale = scale
    self.min = min_val
    self.max = max_val

  def __str__(self):
    return f'BellCurve({self.loc, self.scale, self.min, self.max})'

  def __call__(self, *args, **kwargs):
    assert len(args) == 1 and len(kwargs) == 0
    (x, ) = args
    return numpy.exp(-(x - self.loc) ** 2 / self.scale ** 2) * (self.max - self.min) + self.min


# Maybe shouldn't both having this because it requires this knowledge of the transformation externally
class FixedBellCurveMean(BellCurveMean):
  name = 'fixed-bell-curve'
  DEFAULT_HPARAM_BOUNDS = [
    DEFAULT_BELL_CURVE_LOC_HPARAM_BOUNDS,
    DEFAULT_BELL_CURVE_SCALE_HPARAM_BOUNDS,
  ]

  def __init__(self, *args, **kwargs):
    assert len(args) in (0, 2) and len(kwargs) == 0
    args = args or DEFAULT_FIXED_BELL_CURVE_MEAN_ARGS
    (loc, scale) = args
    min_val, max_val = norm(loc=0, scale=1).ppf([.2, .4])
    super().__init__(loc, scale, min_val, max_val)


# Could definitely clean this up
class GaussianCovariance(object):
  name = 'gaussian'

  def __init__(self, length_scale):
    self.length_scale = length_scale

  def __str__(self):
    return f'Gaussian({self.length_scale})'

  def evaluate_kernel_matrix(self, x_centers, x_eval=None):
    x_eval = x_centers if x_eval is None else x_eval
    distance_matrix_squared = (x_eval[:, None] - x_centers[None, :]) ** 2
    return numpy.exp(-distance_matrix_squared / self.length_scale ** 2)


# Using the idea of b = 1 - length_scale to more closely match the length_scale idea in the Gaussian
# Don't have a great way of thinking about a, but want to leave it floating around, I guess
# It is a potentially very interesting parameter, but I just don't have a perfect sense of it right now
# I also have hard-coded in the [3, 10] domain right now ... would need to deal with that eventually
class CInfinityChebyshevCovariance(object):
  name = 'cinf-cheb'

  def __init__(self, length_scale, parameter_a=1):
    self.length_scale = length_scale
    self.parameter_a = parameter_a

  def __str__(self):
    return f'CInfCheb({self.length_scale}, {self.parameter_a})'

  @staticmethod
  def _rescale(x, min_val, max_val):
    return 2 * (x - min_val) / (max_val - min_val) - 1

  # This assumes that the centers include all the points -- if not true, some other rescaling required
  # Obviously, this could be made more efficient by storing the data -- trying to keep it simple right now
  def evaluate_kernel_matrix(self, x_centers, x_eval=None):
    min_val, max_val = numpy.min(x_centers), numpy.max(x_centers)
    z = self._rescale(x_centers, min_val, max_val)
    x = z if x_eval is None else self._rescale(x_eval, min_val, max_val)
    a = self.parameter_a
    b = 1 - self.length_scale
    x2_z2 = x[:, None] ** 2 + z[None, :] ** 2
    xz = x[:, None] * z[None, :]
    numerator = (b * (1 - b ** 2) - 2 * b * x2_z2 + (1 + 3 * b ** 2) * xz)
    denominator = (1 - b ** 2) ** 2 + 4 * b * (b * x2_z2 - (1 + b ** 2) * xz)
    return 1 - a + 2 * a * (1 - b) * numerator / denominator


# This is only going to work in one dimension -- will consider other stuff eventually, maybe
class GaussianProcess(object):
  def __init__(
    self,
    x,
    y,
    covariance,
    mean_function,
    process_variance,
    noise_variance,
    posterior_regularization=1e-8,
  ):
    self.x = numpy.array(x, dtype=float)
    self.y = numpy.array(y, dtype=float)  # Casting for json (and to avoid integer issues)
    self.covariance = covariance
    self.mean_function = mean_function
    self.process_variance = process_variance
    self.noise_variance = noise_variance
    self.posterior_regularization = posterior_regularization

    self.prior_mean_values = None
    self.kernel_matrix = None
    self.cho_kernel = None
    self.coef = None

    self._setup()

  def _setup(self):
    self.prior_mean_values = self.mean_function(self.x)
    self.kernel_matrix = self.process_variance * self.covariance.evaluate_kernel_matrix(self.x)
    self.cho_kernel = cho_factor(self.kernel_matrix + numpy.diag(self.noise_variance), lower=True, overwrite_a=True)
    self.y_minus_mean = self.y - self.prior_mean_values
    self.coef = cho_solve(self.cho_kernel, self.y_minus_mean)

  @property
  def json_info(self):
    return {
      'y': self.y.tolist(),
      'covariance': str(self.covariance),
      'noise_variance': self.noise_variance.tolist(),
      'process_variance': float(self.process_variance),
      'mean_function': str(self.mean_function),
    }

  @property
  def num_sampled(self):
    return len(self.y)

  # Could reorg normals to not require transpose -- will decide if I care about it eventually
  def posterior_draws(self, num_draws):
    posterior_mean = self.prior_mean_values + numpy.dot(self.kernel_matrix, self.coef)
    temp = solve_triangular(self.cho_kernel[0], self.kernel_matrix.T, lower=self.cho_kernel[1], overwrite_b=False)
    posterior_covariance = self.kernel_matrix - numpy.dot(temp.T, temp)
    posterior_covariance.flat[::self.num_sampled + 1] += self.posterior_regularization
    posterior_cholesky = cholesky(posterior_covariance, lower=True)
    normals = numpy.random.normal(size=(num_draws, self.num_sampled))
    return posterior_mean[None, :] + numpy.dot(normals, posterior_cholesky.T)

  # This isn't the real likelihood, but it's only off by a constant so I don't care about that.
  def compute_log_likelihood(self):
    return (
      -numpy.dot(self.y_minus_mean, cho_solve(self.cho_kernel, self.y_minus_mean)) +
      -2 * numpy.sum(numpy.log(self.cho_kernel[0].diagonal()))
    )
