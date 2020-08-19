import numpy
import matplotlib.pyplot as plt
import tensorfilterkenneth as filterk

from lrtmmse import *

# Simulation Parameters
number_of_antennas = 512
number_of_users = 2
number_of_paths = 4
buffer_size = 16
length = 600
epsilon = 1e-2
user = 0

# Channel Matrix
[channel_matrix, channel_tensor] = filterk.channel(number_of_antennas, number_of_users, number_of_paths, buffer_size)

## Signal QPSK Modulation ##
# signal_0real = 2 * (numpy.random.rand(length, number_of_users) >= 0.5) - 1
# signal_0imag = 2 * (1j * numpy.random.rand(length, number_of_users) >= 0.5) - 1
# signal_0 = signal_0real + signal_0imag

## Signal BPSK Modulation ##
signal_0 = 2 * (numpy.random.rand(length, number_of_users) >= 0.5) - 1

## Obtaining the Training Sequence ##
training = signal_0[:, user]

## Sampling Matrix ##
[matrix_of_sampling, matrix_of_sampling_tensor] = filterk.sampled_matrix(signal_0, number_of_users, number_of_antennas,
                                                                         buffer_size, length)

## Transmited Signal ##
signal_x = channel_matrix @ numpy.transpose(matrix_of_sampling)

## Adding Noise Directly to the Transmited Signal ##
SNR_dB = 30
SNR_Li = 10 ** (SNR_dB / 10)
var_noise = 1 / SNR_Li
shape = signal_x.shape
noise = (np.sqrt(var_noise) / 2) * (numpy.random.randn(*shape) + 1j * numpy.random.randn(*shape))
signal_x = signal_x + noise

## Filtering ##

[filter_coeficients, filter_tensor, signal_x_tensor, errors, iterations] = lrt_mmse(signal_x, training, [8, 8, 8], 3, 6,
                                                                                    epsilon, 500)

plt.figure()
plt.plot(range(0,iterations + 1), errors[0:iterations+1], label='||w_{i+1} - w_{i}||^{2}')
plt.title('Filter Convergence Behavior')
plt.ylabel('Error')
plt.xlabel('Iteration')
plt.legend()
plt.savefig('Filter Convergence Behavior')
plt.show()

print('First 10 Errors:')
print(errors[0:10])
print('Last 10 Errors:')
print(errors[-10:])
print('Number of Iterations Until Convergence:')
print(iterations)

## Filtering the Signal ##

signal_0_approx = numpy.zeros([length]) + 1j * numpy.zeros([length])
for i in range(0, length):
    signal_0_approx[i] = (filter_coeficients.conj().T) @ signal_x[:, i]

## Equalization ##

# signal_0_approxreal = signal_0_approx.real
# signal_0_approxreal[signal_0_approxreal > 0] = 1
# signal_0_approxreal[signal_0_approxreal < 0] = -1
# signal_0_approximag = signal_0_approx.imag
# signal_0_approximag[signal_0_approximag > 0] = 1
# signal_0_approximag[signal_0_approximag < 0] = -1
# signal_0_approx = signal_0_approxreal + 1j * signal_0_approximag

signal_0_approx = signal_0_approx.real
signal_0_approx[signal_0_approx > 0] = 1
signal_0_approx[signal_0_approx < 0] = -1

print('Normalized Mean Square Error:')
print(filterk.normalized_mean_square_error(signal_0[:, user], signal_0_approx))


## Calculating the SINR ##

def sinr1(filter_coeficients, channel_tensor, signal_x, signal_0_approx, number_of_antennas, buffer_size, user):
    Rss = np.var(signal_0_approx) * np.identity(buffer_size)
    Rbb = var_noise * numpy.identity(number_of_antennas)
    Rdd = channel_tensor[user] @ Rss @ channel_tensor[user].conj().T

    Rxx = np.cov(signal_x)
    Rii = Rxx - Rdd - Rbb

    sinr_value = ((filter_coeficients.conj().T @ Rdd) @ filter_coeficients) / (
            (filter_coeficients.conj().T @ (Rii + Rbb)) @ filter_coeficients)

    sinr_value = np.abs(sinr_value)
    sinr_value = 10 * numpy.log10(sinr_value)

    return sinr_value


def sinr2(filter_coeficients, channel_tensor, signal_0_approx, number_of_antennas, buffer_size, number_of_users, user):
    Rss = np.var(signal_0_approx) * np.identity(buffer_size)
    Rbb = var_noise * numpy.identity(number_of_antennas)
    Rdd = channel_tensor[user] @ Rss @ channel_tensor[user].conj().T

    Rii = np.zeros([number_of_antennas, number_of_antennas]) + 1j * np.zeros([number_of_antennas, number_of_antennas])
    for i in range(0, number_of_users):
        if i == user:
            pass
        else:
            Rii = Rii + channel_tensor[i] @ Rss @ channel_tensor[i].conj().T

    sinr_value = ((filter_coeficients.conj().T @ Rdd) @ filter_coeficients) / (
            (filter_coeficients.conj().T @ (Rii + Rbb)) @ filter_coeficients)

    sinr_value = np.abs(sinr_value)
    sinr_value = 10 * numpy.log10(sinr_value)

    return sinr_value


sinr_value1 = sinr1(filter_coeficients, channel_tensor, signal_x, signal_0_approx, number_of_antennas, buffer_size, user)
print('O valor de SINR1 é:')
print(sinr_value1)

sinr_value2 = sinr2(filter_coeficients, channel_tensor, signal_0_approx, number_of_antennas, buffer_size, number_of_users, user)
print('O valor de SINR2 é:')
print(sinr_value2)

