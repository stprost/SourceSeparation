import matplotlib.pyplot as plt

from rnn.src.data import Data
from rnn.src.model import build_model_gru


def plot_loss_acc(history):
    loss = history.history['src1_loss']
    val_loss = history.history['val_src1_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss 1')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()

    loss = history.history['src2_loss']
    val_loss = history.history['val_src2_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss 2')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()

    acc = history.history['src1_accuracy']
    val_acc = history.history['val_src1_accuracy']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy 1')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.clf()

    acc = history.history['src2_accuracy']
    val_acc = history.history['val_src2_accuracy']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy 2')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def train_model(path):
    data_val = Data("dataset/test/m_f")
    mixed_val, src1_val, src2_val = data_val.next_wavs(100)
    mixed_val, src1_val, src2_val = data_val.prepare_data(mixed_val), data_val.prepare_data(
        src1_val), data_val.prepare_data(src2_val)
    data_train = Data("dataset/train/m_f")
    mixed_wav, src1_wav, src2_wav = data_train.next_wavs(-1)
    mixed_batch, src1_batch, src2_batch = data_train.prepare_data(mixed_wav), data_train.prepare_data(
        src1_wav), data_train.prepare_data(src2_wav)
    # mixed_batch = np.loadtxt("dataset/train/mixed_batch.csv", delimiter=',')
    # mixed_batch = mixed_batch.reshape((-1, 4, mixed_batch.shape[2]))
    # src1_batch = np.loadtxt("dataset/train/src1_batch.csv", delimiter=',')
    # src1_batch = src1_batch.reshape((-1, 4, src1_batch.shape[2]))
    # src2_batch = np.loadtxt("dataset/train/src2_batch.csv", delimiter=',')
    # src2_batch = src2_batch.reshape((-1, 4, src2_batch.shape[2]))

    model = build_model_gru()
    model.compile(
        loss="mse",
        optimizer="adam",
        metrics=["accuracy"],
    )

    history = model.fit(
        mixed_batch,
        [src1_batch, src2_batch],
        batch_size=64,
        epochs=10,
        validation_data=(mixed_val, [src1_val, src2_val])
    )
    print(history.history.keys())
    plot_loss_acc(history)
    model.save(path)

    test_scores = model.evaluate(mixed_val, [src1_val, src2_val])
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    print("end")


if __name__ == "__main__":
    train_model("../save/model_1")
