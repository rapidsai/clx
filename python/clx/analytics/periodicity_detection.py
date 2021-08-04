import cupy as cp


def to_periodogram(signal):
    """
    Returns periodogram of signal for finding frequencies that have high energy.

    :param signal: signal (time domain)
    :type signal: cudf.Series
    :return: CuPy array representing periodogram
    :rtype: cupy.ndarray
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

    :param prdg: periodogram to be filtered
    :type signal: cudf.Series
    :param p_value: p-value to filter by
    :type signal: float
    :return: CuPy array representing periodogram
    :rtype: cupy.ndarray
    """

    filtered_prdg = cp.copy(prdg)
    filtered_prdg[filtered_prdg < (cp.mean(filtered_prdg) * (-1) * (cp.log(p_value)))] = 0

    return filtered_prdg


def to_time_domain(prdg):
    """
    Convert the signal back to time domain.

    :param prdg: periodogram (frequency domain)
    :type prdg: cupy.ndarray
    :return: CuPy array representing reconstructed signal
    :rtype: cupy.ndarray
    """

    acf = cp.abs(cp.fft.ifft(prdg))

    return acf
