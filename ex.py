from tensorflow import keras

from data import Data
from model import build_model
from preprocess import to_spectrogram, get_magnitude, spec_to_batch


def prepare_data(wav):
    wav_spec = to_spectrogram(wav)
    wav_mag = get_magnitude(wav_spec)
    wav_batch, _ = spec_to_batch(wav_mag)
    return wav_batch


if __name__ == "__main__":
    data_train = Data("dataset/train/m_f")
    mixed_wav, src1_wav, src2_wav = data_train.next_wavs(1)
    mixed_batch = prepare_data(mixed_wav)
    src1_batch = prepare_data(src1_wav)
    src2_batch = prepare_data(src2_wav)

    model = build_model()
    model.compile(
        loss=keras.losses.MeanSquaredError,
        optimizer="adam",
        metrics=["accuracy"],
    )

    history = model.fit(mixed_batch, src1_batch, batch_size=32, epochs=1, validation_split=0.2)

    # test_scores = model.evaluate(x_test, y_test, verbose=2)
    # print("Test loss:", test_scores[0])
    # print("Test accuracy:", test_scores[1])

    print("kek")
