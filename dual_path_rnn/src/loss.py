import librosa
import tensorflow as tf
import numpy as np


# This file contains methods regarding the SI-SNR loss.
# There is both a numpy and a Tensorflow implementation.

def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def np_sisnr(s, s_hat, do_expand=False, eps=1e-8):
    pad_len = np.abs(s.size - s_hat.size)
    if s.size < s_hat.size:
        s = np.pad(s, (0, pad_len), 'constant', constant_values=0)
    elif s_hat.size < s.size:
        s_hat = np.pad(s_hat, (0, pad_len), 'constant', constant_values=0)
    if do_expand:
        s = np.expand_dims(s, axis=0)
        s_hat = np.expand_dims(s_hat, axis=0)
    dot_product = np.dot(s, s_hat)
    squares = np.dot(s, s)
    s_target = s * dot_product / squares
    e_noise = s_hat - s_target
    s_target_squared = np.dot(squares, squares)
    e_noise_squared = np.dot(e_noise, e_noise)
    return np.log10(s_target_squared / (e_noise_squared + eps))


def sisnr(s, s_hat, do_expand=False, eps=1e-8):
    if do_expand:
        s = np.expand_dims(s, axis=0)
        s_hat = np.expand_dims(s_hat, axis=0)
    dot_product = tf.math.reduce_sum(s * s_hat, axis=1, keepdims=True)
    squares = tf.math.reduce_sum(s * s, axis=1, keepdims=True)
    s_target = s * dot_product / squares
    e_noise = s_hat - s_target
    s_target_squared = tf.math.reduce_sum(s_target * s_target, axis=1)
    e_noise_squared = tf.math.reduce_sum(e_noise * e_noise, axis=1)
    return 10 * log10(s_target_squared / (e_noise_squared + eps))


def permutation_invariant_loss(y_true, y_pred):
    s1 = y_true[:, 0, :]
    s1_hat = y_pred[:, 0, :]
    s2 = y_true[:, 1, :]
    s2_hat = y_pred[:, 1, :]

    sisnr_perm0_spk0 = sisnr(s1, s1_hat)
    sisnr_perm0_spk1 = sisnr(s2, s2_hat)
    sisnr_perm0 = (sisnr_perm0_spk0 + sisnr_perm0_spk1) / 2

    sisnr_perm1_spk0 = sisnr(s1, s2_hat)
    sisnr_perm1_spk1 = sisnr(s2, s1_hat)
    sisnr_perm1 = (sisnr_perm1_spk0 + sisnr_perm1_spk1) / 2

    sisnr_perm_invariant = tf.math.maximum(sisnr_perm0, sisnr_perm1)
    return -sisnr_perm_invariant

    # sisnr_s1 = sisnr(s1, s1_hat) / 10
    # sisnr_s2 = sisnr(s2, s2_hat) / 10
    # res = 1 / (sisnr_s1 + sisnr_s2)
    # return res


def get_permutation_invariant_sisnr(spk0_estimate, spk1_estimate, spk0_groundtruth, spk1_groundtruth):
    perm0_spk0 = sisnr(spk0_groundtruth, spk0_estimate, do_expand=True)
    perm0_spk1 = sisnr(spk1_groundtruth, spk1_estimate, do_expand=True)
    perm1_spk0 = sisnr(spk0_groundtruth, spk1_estimate, do_expand=True)
    perm1_spk1 = sisnr(spk1_groundtruth, spk0_estimate, do_expand=True)

    # Get best permutation
    if perm0_spk0 + perm0_spk1 > perm1_spk0 + perm1_spk1:
        return perm0_spk0, perm0_spk1

    return perm1_spk0, perm1_spk1


if __name__ == "__main__":
    mix, _ = np.array(librosa.load('../ds/tr/mix/MA01_02__FA01_03.wav', sr=8000))
    src1, _ = np.array(librosa.load('../ds/tr/s1/MA01_02__FA01_03.wav', sr=8000))
    src2, _ = np.array(librosa.load('../ds/tr/s2/MA01_02__FA01_03.wav', sr=8000))
    snr1 = np_sisnr(src1, src1)
    snr2 = np_sisnr(src2, src2)
    snr12 = np_sisnr(src1, src2)
    snr21 = np_sisnr(src2, src1)
    snr1m = np_sisnr(src1, mix)
    snr2m = np_sisnr(src2, mix)
    print('kek')
