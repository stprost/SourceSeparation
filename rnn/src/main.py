from keras.utils import CustomObjectScope
import keras
from rnn.src.mask_layer import MaskLayer

from rnn.src.data import Data

if __name__ == "__main__":
    data = Data("dataset/test/m_f")
    mixed_wav, src1_wav, src2_wav = data.next_wavs(100)
    mixed_batch, src1_batch, src2_batch = data.prepare_data(mixed_wav), data.prepare_data(
        src1_wav), data.prepare_data(src2_wav)
    # wav_spec = to_spectrogram(mixed_wav)
    # src_spec = to_spectrogram(src1_wav)
    # wav_phase = get_phase(wav_spec)
    # wav_mag = get_magnitude(wav_spec)
    # mixed_batch, _ = spec_to_batch(wav_spec)

    with CustomObjectScope({'MaskLayer': MaskLayer}):
        model = keras.models.load_model("save/model_1")
    # res = model.predict(mixed_batch)

    # src1_mag = batch_to_spec(res[0], 1)
    # src1_spec = get_stft_matrix(src1_mag, wav_phase)
    # src1_spec = np.reshape(src1_spec, (src1_mag.shape[1], src1_mag.shape[2]))
    # src1 = librosa.istft(src1_spec)
    # write("output/src1_pred.wav", 16000, src1)
    #
    # src2_mag = batch_to_spec(res[1], 1)
    # src2_spec = get_stft_matrix(src2_mag, wav_phase)
    # src2_spec = np.reshape(src2_spec, (src2_mag.shape[1], src2_mag.shape[2]))
    # src2 = librosa.istft(src2_spec)
    # write("output/src2_pred.wav", 16000, src2)
    # print("end")

    test_scores = model.evaluate(mixed_batch, [src1_batch, src2_batch])
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
