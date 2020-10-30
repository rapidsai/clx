import cupy as cp


def to_periodogram(signal):

    # convert cudf series to cupy array
    signal_cp = cp.fromDlpack(signal.to_dlpack())

    # standardize the signal
    signal_cp_std = (signal_cp - cp.mean(signal_cp)) / cp.std(signal_cp)

    # take fourier transform of signal
    FFT_data = cp.fft.fft(signal_cp_std)

    # create periodogram
    prdg = (1 / len(signal)) * ((cp.absolute(FFT_data))**2)

    return prdg


def filter_periodogram(prdg, p_value):

    filtered_prdg = cp.copy(prdg)
    filtered_prdg[filtered_prdg < (cp.mean(filtered_prdg) * (-1) * (cp.log(0.001)))] = 0

    return filtered_prdg


def to_time_domain(prdg):

    acf = cp.abs(cp.fft.ifft(prdg))

    return acf
