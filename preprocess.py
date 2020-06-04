import librosa
import numpy as np
from config import ModelConfig
import soundfile as sf


def get_random_wav(filenames, sec, sr=ModelConfig.SR):
    src1_src2 = np.array(list(
        map(lambda f: _sample_range(_pad_wav(librosa.load(f, sr=sr, mono=False)[0], sr, sec), sr, sec), filenames)))
    mixed = np.array(list(map(lambda f: librosa.to_mono(f), src1_src2)))
    src1, src2 = src1_src2[:, 0], src1_src2[:, 1]
    return mixed, src1, src2


def to_spectrogram(wav, len_frame=ModelConfig.L_FRAME, len_hop=ModelConfig.L_HOP):
    return np.array(list(map(lambda w: librosa.stft(w, n_fft=len_frame, hop_length=len_hop), wav)))


def to_wav(mag, phase, len_hop=ModelConfig.L_HOP):
    stft_matrix = get_stft_matrix(mag, phase)
    return np.array(list(map(lambda s: librosa.istft(s, hop_length=len_hop), stft_matrix)))


def to_wav_from_spec(stft_maxrix, len_hop=ModelConfig.L_HOP):
    return np.array(list(map(lambda s: librosa.istft(s, hop_length=len_hop), stft_maxrix)))


def to_wav_mag_only(mag, init_phase, len_frame=ModelConfig.L_FRAME, len_hop=ModelConfig.L_HOP, num_iters=50):
    # return np.array(list(map(lambda m_p: griffin_lim(m, len_frame, len_hop, num_iters=num_iters, phase_angle=p)[0],
    # list(zip(mag, init_phase))[1])))
    return np.array(list(map(lambda m: lambda p: griffin_lim(m, len_frame, len_hop, num_iters=num_iters, phase_angle=p),
                             list(zip(mag, init_phase))[1])))


def get_magnitude(stft_matrixes):
    return np.abs(stft_matrixes)


def get_phase(stft_maxtrixes):
    return np.angle(stft_maxtrixes)


def get_stft_matrix(magnitudes, phases):
    return magnitudes * np.exp(1.j * phases)


def soft_time_freq_mask(target_src, remaining_src):
    mask = np.abs(target_src) / (np.abs(target_src) + np.abs(remaining_src) + np.finfo(float).eps)
    return mask


def hard_time_freq_mask(target_src, remaining_src):
    mask = np.where(target_src > remaining_src, 1., 0.)
    return mask


def write_wav(data, path, sr=ModelConfig.SR, format='wav', subtype='PCM_16'):
    sf.write('{}.wav'.format(path), data, sr, format=format, subtype=subtype)


def griffin_lim(mag, len_frame, len_hop, num_iters, phase_angle=None, length=None):
    assert (num_iters > 0)
    if phase_angle is None:
        phase_angle = np.pi * np.random.rand(*mag.shape)
    spec = get_stft_matrix(mag, phase_angle)
    for i in range(num_iters):
        wav = librosa.istft(spec, win_length=len_frame, hop_length=len_hop, length=length)
        if i != num_iters - 1:
            spec = librosa.stft(wav, n_fft=len_frame, win_length=len_frame, hop_length=len_hop)
            _, phase = librosa.magphase(spec)
            phase_angle = np.angle(phase)
            spec = get_stft_matrix(mag, phase_angle)
    return wav


def _pad_wav(wav, sr, duration):
    assert (wav.ndim <= 2)

    n_samples = int(sr * duration)
    pad_len = np.maximum(0, n_samples - wav.shape[-1])
    if wav.ndim == 1:
        pad_width = (0, pad_len)
    else:
        pad_width = ((0, 0), (0, pad_len))
    wav = np.pad(wav, pad_width=pad_width, mode='constant', constant_values=0)

    return wav


def _sample_range(wav, sr, duration):
    assert (wav.ndim <= 2)

    target_len = int(sr * duration)
    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - target_len)), 1)[0]
    end = start + target_len
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def batch_to_spec(src, num_wav):
    # shape = (batch_size, n_frames, n_freq) => (batch_size, n_freq, n_frames)
    batch_size, seq_len, freq = src.shape
    src = np.reshape(src, (num_wav, -1, freq))
    src = src.transpose(0, 2, 1)
    return src

def spec_to_batch(src):
        num_wavs, freq, n_frames = src.shape

        # Padding
        pad_len = 0
        if n_frames % ModelConfig.SEQ_LEN > 0:
            pad_len = (ModelConfig.SEQ_LEN - (n_frames % ModelConfig.SEQ_LEN))
        pad_width = ((0, 0), (0, 0), (0, pad_len))
        padded_src = np.pad(src, pad_width=pad_width, mode='constant', constant_values=0)

        assert(padded_src.shape[-1] % ModelConfig.SEQ_LEN == 0)

        batch = np.reshape(padded_src.transpose(0, 2, 1), (-1, ModelConfig.SEQ_LEN, freq))
        return batch, padded_src