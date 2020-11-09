import cupy as cp


def to_periodogram(signal):
    """
    Returns periodogram of signal for finding frequencies that have high energy.

    :param signal: signal (time domain)
    :type signal: cudf.Series
    :return: CuPy array representing periodogram
    :rtype: cupy.core.core.ndarray
    """

    # convert cudf series to cupy array
    signal_cp = cp.fromDlpack(signal.to_dlpack())

    # standardize the signal
    signal_cp_std = (signal_cp - cp.mean(signal_cp)) / cp.std(signal_cp)

    # take fourier transform of signal
    FFT_data = cp.fft.fft(signal_cp_std)

    # create periodogram
    prdg = (1 / len(signal)) * ((cp.absolute(FFT_data)) ** 2)

    return prdg


def filter_periodogram(prdg, p_value):
    """
    Select important frequencies by filtering periodogram by p-value. Filtered out frequencies are set to zero.

    :param periodogram: periodogram to be filtered
    :type signal: cudf.Series
    :return: CuPy array representing periodogram
    :rtype: cupy.core.core.ndarray
    """

    filtered_prdg = cp.copy(prdg)
    filtered_prdg[filtered_prdg < (cp.mean(filtered_prdg) * (-1) * (cp.log(0.001)))] = 0

    return filtered_prdg


def to_time_domain(prdg):
    """
    Convert periodogram to signal in time domain.

    :param periodogram: periodogram (frequency domain)
    :type signal: cupy.core.core.ndarray
    :return: CuPy array representing reconstructed signal
    :rtype: cupy.core.core.ndarray
    """

    acf = cp.abs(cp.fft.ifft(prdg))

    return acf
