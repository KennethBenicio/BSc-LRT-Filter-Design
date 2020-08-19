# Imports
import numpy as np
import tensorly as tl

# Usefull functions

# Correlation Coeficient
def corrcoef(vector1, vector2):
    product = np.mean((vector1 - vector1.mean()) * (vector2 - vector2.mean()))
    stds = vector1.std() * vector2.std()

    if stds == 0:
        return 0
    else:
        product = product / stds

        return product


# Auto-Correlation Matrix
def autocorrmatrix(matrix):
    if np.isreal(matrix).all():
        autocorr_matrix = np.zeros([matrix.shape[1], matrix.shape[1]])
    else:
        autocorr_matrix = np.zeros([matrix.shape[1], matrix.shape[1]], dtype='complex')

    for i in range(0, matrix.shape[1]):
        for j in range(0, matrix.shape[1]):
            autocorr_matrix[i, j] = correlation_coefficient(matrix[:, i], matrix[:, j])

    return autocorr_matrix


# Cross-Correlation Matrix
def corrmatrix(matrix1, matrix2):
    if np.isreal(matrix1).all() and np.isreal(matrix2).all():
        corr_matrix = np.zeros([matrix1.shape[1], matrix1.shape[1]])
    else:
        corr_matrix = np.zeros([matrix1.shape[1], matrix1.shape[1]], dtype='complex')

    for i in range(0, matrix1.shape[1]):
        for j in range(0, matrix1.shape[1]):
            corr_matrix[i, j] = correlation_coefficient(matrix1[:, i], matrix2[:, j])

    return corr_matrix


def normalized_mean_square_error(X, X_hat):
    nmse = (np.linalg.norm(X - X_hat)) ** 2 / (np.linalg.norm(X)) ** 2
    return nmse


# Linear Filters

def wiener(received_signal, training_sequence):
    Rxx = received_signal @ (received_signal.conj().T) / received_signal.shape[1]
    Rxs = received_signal @ (training_sequence.conj()) / received_signal.shape[1]
    W = (np.linalg.inv(Rxx) @ Rxs)

    training_sequence_approx = W.T @ received_signal
    nmse = normalized_mean_square_error(training_sequence, training_sequence_approx)

    return training_sequence_approx, nmse


def lmmse(X, Y):
    X_approx = ((np.cov(X.T, Y.T)[0][1]) / np.var(Y)) * (Y - np.mean(Y)) + np.mean(X)
    NMSE = normalized_mean_square_error(X, X_approx)

    return X_approx, NMSE


def lms(signal_corrupted, signal, step, order):
    # Number of loops to count because of the buffer delay.
    I = signal_corrupted.size - order + 1
    # Filter output.
    signal_approx = np.zeros([I, 1], dtype='complex')
    # Error vector.
    error = np.zeros((I, 1), dtype='complex')
    # Matrix that will store the filter coeficients for each loop.
    w = np.zeros((order, I), dtype='complex')

    mse = np.zeros(I, )
    for i in range(I):
        u = np.flipud(signal_corrupted[i:i + order])[:, np.newaxis]
        signal_approx[i] = w[:, [i - 1]].T.conj() @ u
        error[i] = signal[i + order - 1] - signal_approx[i]
        mse[i] = np.absolute(error[i]) ** 2 / I

        # Updating the filter coeficients
        w[:, [i]] = w[:, [i - 1]] + step * u * (error[i].conj())
        signal_approx[i] = w[:, [i]].T.conj() @ u

    return mse


def nlms(signal_corrupted, signal, step, order):
    # Number of loops to count because of the buffer delay.
    I = signal_corrupted.size - order + 1
    # Filter output.
    signal_approx = np.zeros([I, 1], dtype='complex')
    # Error vector.
    error = np.zeros((I, 1), dtype='complex')
    # Matrix that will store the filter coeficients for each loop.
    w = np.zeros((order, I), dtype='complex')

    mse = np.zeros(I, )
    for i in range(I):
        u = np.flipud(signal_corrupted[i:i + order])[:, np.newaxis]
        signal_approx[i] = w[:, [i - 1]].T.conj() @ u
        error[i] = signal[i + order - 1] - signal_approx[i]
        mse[i] = np.absolute(error[i]) ** 2 / I

        # Updating the filter coeficients
        w[:, [i]] = w[:, [i - 1]] + (step / np.linalg.norm(u)) * u * (error[i].conj())
        signal_approx[i] = w[:, [i]].T.conj() @ u

    return mse


# Multilinear Filters

def tlms(signal_corrupted, signal, step):
    _, sample = signal_corrupted.shape
    nh = 10
    nv = 10
    w_h = np.zeros([nh, 1])
    w_h[0] = 1
    w_v = np.zeros([nv, 1])
    w_v[0] = 1

    mse = np.zeros(sample)
    for i in range(0, sample):
        x = signal_corrupted[:, i]
        x_matrix = x.reshape([nh, nv], order='F')
        u_h = x_matrix @ (w_v.conj())
        u_v = x_matrix.T @ (w_h.conj())
        error = signal[i,] - (tl.tenalg.kronecker([w_v, w_h]).T.conj()) @ x
        step_approx = step / (np.linalg.norm(u_h, 2) ** 2 + np.linalg.norm(u_v, 2) ** 2)
        w_h = w_h + step_approx * u_h * error.conj()
        w_v = w_v + step_approx * u_v * error.conj()
        w = tl.tenalg.kronecker([w_v, w_h])
        mse[i] = ((np.absolute(signal[i,] - w.T.conj() @ x)) ** 2) / sample

    return w, mse


def atlms(signal_corrupted, signal, step):
    _, sample = signal_corrupted.shape

    Kh = 10
    Kv = 10
    Kb = np.floor(sample / (Kh + Kv))

    w_h = np.zeros([Kh, 1])
    w_h[0] = 1
    w_v = np.zeros([Kv, 1])
    w_v[0] = 1

    z = 0
    mse = np.zeros(sample)
    for i in range(0, int(Kb * (Kh + Kv)), int(Kh + Kv)):

        for j in range(i, int(i + Kh)):
            x = signal_corrupted[:, j]
            x_matrix = x.reshape([Kh, Kv], order='F')
            u_h = x_matrix @ (w_v.conj())
            error = signal[j,] - (tl.tenalg.kronecker([w_v, w_h]).T.conj()) @ x
            step_approx = step / (np.linalg.norm(u_h, 2) ** 2)
            w_h = w_h + step_approx * u_h * error.conj()
            w = tl.tenalg.kronecker([w_v, w_h])
            mse[z] = np.absolute(error) ** 2 / sample
            z = z + 1

        for k in range(int(i + Kh), int(i + Kh + Kv)):
            x = signal_corrupted[:, k]
            x_matrix = x.reshape([Kh, Kv], order='F')
            u_v = x_matrix.T @ (w_h.conj())
            error = signal[k,] - (tl.tenalg.kronecker([w_v, w_h]).T.conj()) @ x
            step_approx = step / (np.linalg.norm(u_v, 2) ** 2)
            w_v = w_v + step_approx * u_v * error.conj()
            w = tl.tenalg.kronecker([w_v, w_h])
            mse[z] = np.absolute(error) ** 2 / sample
            z = z + 1

    return w, mse


# Functions for simulation of multi-path MU-MIMO channel model.

def channel(number_of_antenas, number_of_users, number_of_paths, buffer_size):
    # Wavelength of the Carrier.
    l = 100
    # Speed of light
    c = 3 ** 8
    # Symbol Period.
    T = 10 ** -1
    # Wavelength of the Carrier.

    channel_gain = np.random.randn(number_of_users, number_of_paths) / np.sqrt(2) + 1j * np.random.randn(
        number_of_users, number_of_paths) / np.sqrt(2)
    channel_gain = (channel_gain - np.mean(channel_gain)) / np.std(channel_gain)

    angles_of_arrival = -np.pi / 2 + np.pi * np.random.rand(number_of_users, number_of_paths)
    channel_spatial_response = np.zeros([number_of_users, number_of_paths, number_of_antenas], dtype='complex')
    for i in range(0, number_of_users):
        for j in range(0, number_of_paths):
            for k in range(0, number_of_antenas):
                channel_spatial_response[i, j, k] = np.exp(-1j * np.pi * (k) * np.cos(angles_of_arrival[i, j]))

    delay_response = l / (2 * c * np.cos(angles_of_arrival))
    pulse_shaping_waveform = np.zeros([number_of_users, number_of_paths, buffer_size], dtype='complex')
    for i in range(0, number_of_users):
        for j in range(0, number_of_paths):
            for k in range(0, buffer_size):
                pulse_shaping_waveform[i, j, k] = np.sinc(i * T - delay_response[i, j]) + 1j * np.sinc(
                    k * T - delay_response[i, j])

    channel_tensor = np.zeros([number_of_users, number_of_paths, number_of_antenas, buffer_size], dtype='complex')
    for i in range(0, number_of_users):
        for j in range(0, number_of_paths):
            csr = (channel_spatial_response[i, j])[:, None]
            psw = (pulse_shaping_waveform[i, j])[:, None]
            channel_tensor[i, j] = (channel_gain[i, j] * csr) @ psw.T

    channel_tensor = np.sum(channel_tensor, 1)
    channel_matrix = tl.unfold(channel_tensor, 1)

    return channel_matrix, channel_tensor


def sampled_signal(user_signal, number_of_users, buffer_size, instant_of_sampling):
    user_signal_at_k = np.zeros([number_of_users, buffer_size], dtype='complex')

    # This block simulates the change of signal throughout time in the buffer of data.
    for i in range(0, number_of_users):
        signal = user_signal[:, i]
        signal = signal[:, np.newaxis]
        signal = np.append(signal, np.zeros([1, buffer_size]))
        signal = signal[instant_of_sampling:instant_of_sampling + buffer_size]
        signal = (signal[::-1])
        user_signal_at_k[i, :] = signal

    return user_signal_at_k


def sampled_matrix(user_signal, number_of_users, number_of_antenas, buffer_size, lenght):
    matrix_of_sampling = tl.tensor(np.zeros([lenght, buffer_size, number_of_users], dtype='complex'))

    for j in range(0, number_of_users):
        for i in range(0, lenght):
            user_at_instant_i = sampled_signal(user_signal, number_of_users, buffer_size, i)
            matrix_of_sampling[i, :, j] = user_at_instant_i[j, :]

    matrix_of_sampling = np.transpose(matrix_of_sampling)
    matrix_of_sampling_tensor = matrix_of_sampling
    matrix_of_sampling = tl.unfold(matrix_of_sampling, 2)

    return matrix_of_sampling, matrix_of_sampling_tensor


def noise(received_signal_uncorrupted, snr_dB):
    snr_linear = 10 ** (snr_dB / 10)
    b = (1 /(2 * snr_linear)) * (np.random.randn(*received_signal_uncorrupted.shape) + np.random.randn(
        *received_signal_uncorrupted.shape))
    received_signal_corrupted = received_signal_uncorrupted + b

    return received_signal_corrupted, np.var(b)


def received_signal(matrix_of_sampling, channel_matrix, snr_dB):
    received_signal_uncorrupted = channel_matrix @ np.transpose(matrix_of_sampling)
    [received_signal_corrupted, var_noise] = noise(received_signal_uncorrupted, snr_dB)

    return received_signal_uncorrupted, received_signal_corrupted, var_noise