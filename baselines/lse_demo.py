from scipy.special import gamma as gamma_function
import numpy

deadhead_means = (7.2, 5.4, 8)
deadhead_variances = (0.9876, .2, 5)
deadhead_times = numpy.arange(3, 11)
hist_buckets = numpy.arange(3, 12) - .5

LSE_NAME = 'lse'

# Shouldn't actually use this, just copying it here for a moment

max_calls = 50
num_tries = 200

def run_one_lse_baseline_trial(deadhead_simulator):
  pass

# for deadhead_mean, deadhead_variance in zip(deadhead_means, deadhead_variances):
#   beta = deadhead_mean / deadhead_variance
#   alpha = deadhead_mean * beta
#   mode = (alpha - 1) / beta
#
#
#   def yf_unnorm(x):  # Should log space for safety??
#     return beta ** alpha / gamma_function(alpha) * x ** (alpha - 1) * numpy.exp(-beta * x)
#
#
#   deadhead_distribution = yf_unnorm(deadhead_times) / yf_unnorm(mode) * .2 + .2
#
#   wins = []
#   gaps = []
#   for now in range(num_tries):
#     tries = numpy.zeros_like(deadhead_times)
#     successes = numpy.zeros_like(deadhead_times)
#     calls = numpy.random.random(max_calls)
#
#     indices_under_consideration = [0, 3, 5, 7]  # The first step of the LSE
#     initial_sampling_count = 8
#     call_count = 0
#     for this_index in indices_under_consideration:
#       for _ in range(initial_sampling_count):
#         prob_of_success = deadhead_distribution[this_index]
#         successes[this_index] += prob_of_success < calls[call_count]
#         call_count += 1
#         tries[this_index] += 1
#     results = successes / (tries + 1e-10)
#     winning_indices = numpy.where(results == max(results))[0]
#     winning_index = numpy.random.choice(winning_indices)
#
#     # The second step of the LSE
#     if winning_index in (0, 3):
#       second_sampling_count = 8
#       indices_under_consideration = [2]
#       for this_index in indices_under_consideration:
#         for _ in range(second_sampling_count):
#           prob_of_success = deadhead_distribution[this_index]
#           successes[this_index] += prob_of_success < calls[call_count]
#           call_count += 1
#           tries[this_index] += 1
#       results = successes / (tries + 1e-10)
#       winning_indices = numpy.where(results == max(results))[0]
#       winning_index = numpy.random.choice(winning_indices)
#
#       if winning_index in (0, 2):
#         third_sampling_count = 8
#         indices_under_consideration = [1]
#         for this_index in indices_under_consideration:
#           for _ in range(third_sampling_count):
#             prob_of_success = deadhead_distribution[this_index]
#             successes[this_index] += prob_of_success < calls[call_count]
#             call_count += 1
#             tries[this_index] += 1
#         final_indices_for_polish = [0, 1, 2, 3]
#       else:
#         third_sampling_count = 8
#         indices_under_consideration = [4]
#         for this_index in indices_under_consideration:
#           for _ in range(third_sampling_count):
#             prob_of_success = deadhead_distribution[this_index]
#             successes[this_index] += prob_of_success < calls[call_count]
#             call_count += 1
#             tries[this_index] += 1
#         final_indices_for_polish = [3, 4, 5]
#       while call_count < max_calls:  # Final polish
#         this_index = numpy.random.choice(final_indices_for_polish)
#         prob_of_success = deadhead_distribution[this_index]
#         successes[this_index] += prob_of_success < calls[call_count]
#         call_count += 1
#         tries[this_index] += 1
#     else:
#       second_sampling_count = 8  # We have only 5 points to consider so we can get ten per time
#       indices_under_consideration = [4, 6]
#       for this_index in indices_under_consideration:
#         for _ in range(second_sampling_count):
#           prob_of_success = deadhead_distribution[this_index]
#           successes[this_index] += prob_of_success < calls[call_count]
#           call_count += 1
#           tries[this_index] += 1
#       indices_under_consideration = [3, 4, 5, 6, 7]
#       while call_count < max_calls:  # Final polish
#         this_index = numpy.random.choice(indices_under_consideration)
#         prob_of_success = deadhead_distribution[this_index]
#         successes[this_index] += prob_of_success < calls[call_count]
#         call_count += 1
#         tries[this_index] += 1
#
#     assert call_count == max_calls
#
#     results = successes / (tries + 1e-10)
#     winning_indices = numpy.where(results == max(results))[0]
#     winning_index = numpy.random.choice(winning_indices)
#     wins.append(deadhead_times[winning_index])
#     gaps.append(max(deadhead_distribution) - deadhead_distribution[winning_index])
#
#   prob_of_results = numpy.histogram(wins, hist_buckets)[0] / num_tries
#   gap = max(deadhead_distribution) - deadhead_distribution
#   print(sum(prob_of_results * gap), numpy.mean(gaps), numpy.std(gaps), numpy.percentile(gaps, 25),
#         numpy.percentile(gaps, 75))
