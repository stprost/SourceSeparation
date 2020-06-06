from data import Data
import librosa
import numpy as np
from scipy.io.wavfile import write
from keras.utils import CustomObjectScope
import keras
from mask_layer import MaskLayer

from data import Data
from preprocess import batch_to_spec, to_spectrogram, get_magnitude, spec_to_batch, get_phase, to_wav, get_stft_matrix

if __name__ == "__main__":
    data = Data("dataset/test/m_f")
    mixed_wav, src1_wav, src2_wav = data.next_wavs(1)
    wav_spec = to_spectrogram(mixed_wav)
    src_spec = to_spectrogram(src1_wav)
    wav_phase = get_phase(wav_spec)
    wav_mag = get_magnitude(wav_spec)
    mixed_batch, _ = spec_to_batch(wav_spec)

    # wav_spec = to_spectrogram(src1_wav)
    # wav_phase = get_phase(wav_spec)
    # wav_mag = get_magnitude(wav_spec)
    # wav_batch, _ = spec_to_batch(wav_mag)
    #
    # src1_mag = batch_to_spec(wav_batch, 1)
    # src1 = to_wav(src1_mag, wav_phase)
    # src1_spec = get_stft_matrix(src1_mag, wav_phase)
    # src1_spec = np.reshape(src1_spec, (src1_mag.shape[1], src1_mag.shape[2]))
    # src1 = librosa.istft(src1_spec)
    # write("output/src1.wav", 16000, src1)

    with CustomObjectScope({'MaskLayer': MaskLayer}):
        model = keras.models.load_model("save/model_1")
    res = model.predict(mixed_batch)
    src1_mag = batch_to_spec(res, 1)
    src1_spec = get_stft_matrix(src1_mag, wav_phase)
    src1_spec = np.reshape(src1_spec, (src1_mag.shape[1], src1_mag.shape[2]))
    src1 = librosa.istft(src1_spec)
    write("output/src1.wav", 16000, src1)
    print("end")
    # test_scores = model.evaluate(x_test, y_test, verbose=2)
    # print("Test loss:", test_scores[0])
    # print("Test accuracy:", test_scores[1])
