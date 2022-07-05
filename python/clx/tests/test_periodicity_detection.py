import cudf
import cupy as cp
from clx.analytics import periodicity_detection as pdd


def test_to_periodogram():

    expected_periodogram = cp.array([
        2.14782297e-30,
        8.83086404e-02,
        5.23325583e-02,
        1.99054116e-01,
        9.58452790e-01,
        5.40114641e00,
        1.04142106e00,
        2.46821568e-01,
        5.06332729e-02,
        3.44875313e-02,
        2.97378597e-01,
        7.47935264e-02,
        3.87987331e-02,
        6.56637625e-02,
        1.34893777e-01,
        1.13015864e00,
        7.79747216e-03,
        1.14757856e-01,
        4.10151947e-01,
        2.84306210e-01,
        1.25890800e-02,
        2.56152419e-01,
        4.40248947e-01,
        2.64140790e-01,
        8.26499055e-01,
        5.82104062e-01,
        2.04041628e00,
        4.24631265e00,
        1.53295952e-01,
        1.36986604e00,
        6.93053951e00,
        3.77611060e00,
        3.79886075e00,
        4.40471582e-01,
        3.98427502e-01,
        8.63914848e00,
        1.13520190e-01,
        7.77541742e-01,
        1.65678473e00,
        1.60364982e00,
        2.53134486e00,
        4.42140629e-01,
        1.15635914e-01,
        7.41331357e-01,
        1.91152360e-01,
        1.17622857e-01,
        2.08266982e-01,
        2.38361680e-02,
        1.18239068e00,
        1.03731817e00,
        1.29349009e-01,
        1.28179689e00,
        1.91976049e-01,
        1.17875358e-01,
        1.10296708e-01,
        7.84909233e-01,
        1.34339221e-01,
        6.32343429e-02,
        8.14424044e-01,
        3.22720512e-01,
        3.22720512e-01,
        8.14424044e-01,
        6.32343429e-02,
        1.34339221e-01,
        7.84909233e-01,
        1.10296708e-01,
        1.17875358e-01,
        1.91976049e-01,
        1.28179689e00,
        1.29349009e-01,
        1.03731817e00,
        1.18239068e00,
        2.38361680e-02,
        2.08266982e-01,
        1.17622857e-01,
        1.91152360e-01,
        7.41331357e-01,
        1.15635914e-01,
        4.42140629e-01,
        2.53134486e00,
        1.60364982e00,
        1.65678473e00,
        7.77541742e-01,
        1.13520190e-01,
        8.63914848e00,
        3.98427502e-01,
        4.40471582e-01,
        3.79886075e00,
        3.77611060e00,
        6.93053951e00,
        1.36986604e00,
        1.53295952e-01,
        4.24631265e00,
        2.04041628e00,
        5.82104062e-01,
        8.26499055e-01,
        2.64140790e-01,
        4.40248947e-01,
        2.56152419e-01,
        1.25890800e-02,
        2.84306210e-01,
        4.10151947e-01,
        1.14757856e-01,
        7.79747216e-03,
        1.13015864e00,
        1.34893777e-01,
        6.56637625e-02,
        3.87987331e-02,
        7.47935264e-02,
        2.97378597e-01,
        3.44875313e-02,
        5.06332729e-02,
        2.46821568e-01,
        1.04142106e00,
        5.40114641e00,
        9.58452790e-01,
        1.99054116e-01,
        5.23325583e-02,
        8.83086404e-02,
    ])

    signal = cudf.Series([
        3274342,
        3426017,
        3758781,
        3050763,
        3765678,
        3864117,
        3287878,
        3397645,
        3509973,
        3844070,
        3725934,
        3287715,
        3373505,
        3909898,
        3630503,
        3070180,
        3528452,
        3801183,
        3277141,
        3625685,
        3142354,
        3140470,
        3829668,
        3623178,
        3129990,
        3549270,
        3928100,
        3331894,
        3599137,
        3978103,
        3471284,
        3220011,
        3654968,
        3789411,
        3584702,
        3512986,
        3401678,
        3774912,
        3461276,
        3549195,
        3320150,
        3655766,
        3562267,
        3525937,
        3267010,
        3441179,
        3596828,
        3208453,
        3167370,
        4036471,
        3358863,
        3169950,
        3341009,
        4010556,
        3317385,
        3132360,
        3753407,
        3808679,
        3499711,
        3248874,
        3945531,
        3837029,
        3400068,
        3625813,
        3612960,
        3523530,
        3427957,
        3749848,
        3475452,
        3289964,
        3238560,
        3428817,
        3489523,
        3429917,
        3557773,
        3432514,
        3459938,
        3440332,
        3296710,
        3711087,
        3729805,
        3447954,
        3773181,
        3855161,
        3955022,
        3252652,
        3599792,
        3769181,
        3809061,
        3495044,
        3396623,
        3680456,
        3358306,
        3368779,
        3469016,
        3169477,
        3449529,
        3738450,
        3293116,
        3303107,
        3522923,
        3746871,
        3436093,
        3124102,
        3679797,
        3829441,
        3641894,
        3654410,
        3588528,
        3628979,
        3738718,
        3737379,
        3370349,
        3583376,
        3694398,
        3559319,
        3464402,
        3421738,
        3265208,
    ])

    actual_periodgram = pdd.to_periodogram(signal)

    assert cp.allclose(actual_periodgram, expected_periodogram)


def test_filter_periodogram():

    periodogram = cp.array([
        2.14782297e-30,
        8.83086404e-02,
        5.23325583e-02,
        1.99054116e-01,
        9.58452790e-01,
        5.40114641e00,
        1.04142106e00,
        2.46821568e-01,
        5.06332729e-02,
        3.44875313e-02,
        2.97378597e-01,
        7.47935264e-02,
        3.87987331e-02,
        6.56637625e-02,
        1.34893777e-01,
        1.13015864e00,
        7.79747216e-03,
        1.14757856e-01,
        4.10151947e-01,
        2.84306210e-01,
        1.25890800e-02,
        2.56152419e-01,
        4.40248947e-01,
        2.64140790e-01,
        8.26499055e-01,
        5.82104062e-01,
        2.04041628e00,
        4.24631265e00,
        1.53295952e-01,
        1.36986604e00,
        6.93053951e00,
        3.77611060e00,
        3.79886075e00,
        4.40471582e-01,
        3.98427502e-01,
        8.63914848e00,
        1.13520190e-01,
        7.77541742e-01,
        1.65678473e00,
        1.60364982e00,
        2.53134486e00,
        4.42140629e-01,
        1.15635914e-01,
        7.41331357e-01,
        1.91152360e-01,
        1.17622857e-01,
        2.08266982e-01,
        2.38361680e-02,
        1.18239068e00,
        1.03731817e00,
        1.29349009e-01,
        1.28179689e00,
        1.91976049e-01,
        1.17875358e-01,
        1.10296708e-01,
        7.84909233e-01,
        1.34339221e-01,
        6.32343429e-02,
        8.14424044e-01,
    ])

    expected_filtered = cp.array([
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        6.93053951,
        0.0,
        0.0,
        0.0,
        0.0,
        8.63914848,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    actual_filtered = pdd.filter_periodogram(periodogram, 0.001)

    assert cp.allclose(actual_filtered, expected_filtered)


def test_to_domain():

    periodogram = cp.array([
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        6.93053951,
        0.0,
        0.0,
        0.0,
        0.0,
        8.63914848,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    expected_signal = cp.array([
        0.26389302,
        0.25470949,
        0.22783294,
        0.18526557,
        0.13035481,
        0.06865704,
        0.02978938,
        0.08107234,
        0.14211096,
        0.19489424,
        0.23454643,
        0.25800256,
        0.26352351,
        0.25070444,
        0.22048766,
        0.17513786,
        0.11830457,
        0.05657488,
        0.03572989,
        0.09357597,
        0.15352238,
        0.20399289,
        0.24060791,
        0.26057394,
        0.26241606,
        0.24599927,
        0.21253286,
        0.16454466,
        0.10602009,
        0.04528412,
        0.04528412,
        0.10602009,
        0.16454466,
        0.21253286,
        0.24599927,
        0.26241606,
        0.26057394,
        0.24060791,
        0.20399289,
        0.15352238,
        0.09357597,
        0.03572989,
        0.05657488,
        0.11830457,
        0.17513786,
        0.22048766,
        0.25070444,
        0.26352351,
        0.25800256,
        0.23454643,
        0.19489424,
        0.14211096,
        0.08107234,
        0.02978938,
        0.06865704,
        0.13035481,
        0.18526557,
        0.22783294,
        0.25470949,
    ])

    actual_signal = pdd.to_time_domain(periodogram)

    assert cp.allclose(actual_signal, expected_signal)
