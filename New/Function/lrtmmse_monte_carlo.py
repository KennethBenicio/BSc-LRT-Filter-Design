import numpy as np
import matplotlib.pyplot as plt
import tensorfilterkenneth as filterk

from tqdm import tqdm
from lrtmmse import *

# Simulation Parameters
number_of_antennas = 512
number_of_users = 4
number_of_paths = 5
buffer_size = 16
length = 600
epsilon = 1e-2
user = 0


def sinr1(filter_coeficients, channel_tensor, signal_x, signal_0_approx, number_of_antennas, buffer_size, user):
    Rss = np.var(signal_0_approx) * np.identity(buffer_size)
    Rbb = var_noise * np.identity(number_of_antennas)
    Rdd = channel_tensor[user] @ Rss @ channel_tensor[user].conj().T

    Rxx = np.cov(signal_x)
    Rii = Rxx - Rdd - Rbb

    sinr_value = ((filter_coeficients.conj().T @ Rdd) @ filter_coeficients) / (
            (filter_coeficients.conj().T @ (Rii + Rbb)) @ filter_coeficients)

    sinr_value = np.abs(sinr_value)
    sinr_value = 10 * np.log10(sinr_value)

    return sinr_value

def sinr2(filter_coeficients, channel_tensor, signal_0_approx, number_of_antennas, buffer_size, number_of_users, user):
    Rss = np.var(signal_0_approx) * np.identity(buffer_size)
    Rbb = var_noise * np.identity(number_of_antennas)
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
    sinr_value = 10 * np.log10(sinr_value)

    return sinr_value

run = 500
nmse = np.zeros([4, ])
sinr_value1 = np.zeros([4, ])
sinr_value2 = np.zeros([4, ])
snr_dB = np.array([0, 10, 20, 30])
for i in tqdm(range(run)):
    for j in tqdm(range(len(snr_dB))):
        # Channel Matrix
        [channel_matrix, channel_tensor] = filterk.channel(number_of_antennas, number_of_users, number_of_paths,
                                                           buffer_size)

        ## Signal QPSK Modulation ##
        # signal_0real = 2 * (numpy.random.rand(length, number_of_users) >= 0.5) - 1
        # signal_0imag = 2 * (1j * numpy.random.rand(length, number_of_users) >= 0.5) - 1
        # signal_0 = signal_0real + signal_0imag

        ## Signal BPSK Modulation ##
        signal_0 = 2 * (np.random.rand(length, number_of_users) >= 0.5) - 1

        ## Obtaining the Training Sequence ##
        training = signal_0[:, 0]

        ## Sampling Matrix of Signal ##
        [matrix_of_sampling, matrix_of_sampling_tensor] = filterk.sampled_matrix(signal_0, number_of_users,
                                                                                 number_of_antennas, buffer_size,
                                                                                 length)

        ## Transmited Signal ##
        signal_x = channel_matrix @ np.transpose(matrix_of_sampling)

        ## Adding Noise Directly to the Transmited Signal ##
        SNR_dB = 30
        SNR_Li = 10 ** (SNR_dB / 10)
        var_noise = 1 / SNR_Li
        shape = signal_x.shape
        noise = (np.sqrt(var_noise) / 2) * (np.random.randn(*shape) + 1j * np.random.randn(*shape))
        signal_x = signal_x + noise

        ## Filtering ##
        [filter_coeficients, filter_tensor, signal_x_tensor, errors, iterations] = lrt_mmse(signal_x, training,
                                                                                            [8, 8, 8], 3, 3, epsilon, 500)

        ## Filtering the Signal ##
        signal_0_approx = np.zeros([length]) + 1j * np.zeros([length])
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

        nmse[j] = nmse[j] + filterk.normalized_mean_square_error(signal_0[:, user], signal_0_approx)

        sinr_value1[j] = sinr_value1[j] + sinr1(filter_coeficients, channel_tensor, signal_x, signal_0_approx, number_of_antennas, buffer_size, user)
        sinr_value2[j] = sinr_value2[j] + sinr2(filter_coeficients, channel_tensor, signal_0_approx, number_of_antennas, buffer_size, number_of_users, user)

nmse = nmse / run
sinr_value1 = sinr_value1 / run
sinr_value2 = sinr_value2 / run

plt.figure()
plt.semilogy(snr_dB, nmse, label='nmse')
plt.title('NMSE Error Behavior')
plt.ylabel('NMSE Error')
plt.xlabel('SNR(dB)')
plt.legend()
plt.savefig('NMSE Error Behavior in LRTMMSE')
plt.show()

plt.figure()
plt.plot(snr_dB, sinr_value1, label='SINR(dB)')
plt.title('SINR Behavior')
plt.ylabel('SINR(dB)')
plt.xlabel('SNR(dB)')
plt.legend()
plt.savefig('SINR1 Behavior in LRTMMSE')
plt.show()

plt.figure()
plt.plot(snr_dB, sinr_value2, label='SINR(dB)')
plt.title('SINR Behavior')
plt.ylabel('SINR(dB)')
plt.xlabel('SNR(dB)')
plt.legend()
plt.savefig('SINR2 Behavior in LRTMMSE')
plt.show()