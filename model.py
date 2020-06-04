import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from config import ModelConfig


def build_model():
    inputs = keras.Input(shape=(None, ModelConfig.L_FRAME // 2 + 1))

    src = layers.Dense(np.shape(inputs)[2], activation="relu")(inputs)

    # rnn_cells = [layers.GRUCell(ModelConfig.HID_SIZE) for _ in range(ModelConfig.NUM_LAYERS)]
    # stacked_gru = layers.StackedRNNCells(rnn_cells)
    # output_rnn = layers.RNN(stacked_gru)(inputs)
    # input_size = np.shape(inputs)[2]
    # src1_pre = layers.Dense(input_size, activation="relu")(output_rnn)
    # src2_pre = layers.Dense(input_size, activation="relu")(output_rnn)
    #
    # # time-freq masking layer
    # src1 = src1_pre / (src1_pre + src2_pre + np.finfo(float).eps) * inputs
    # src2 = src2_pre / (src1_pre + src2_pre + np.finfo(float).eps) * inputs

    model = keras.Model(inputs=inputs, outputs=src, name="gru_model")
    model.summary()
    return model

