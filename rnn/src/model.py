import numpy as np
import keras
from rnn.src.mask_layer import MaskLayer
from keras import layers

from rnn.src.config import ModelConfig


def build_model_gru_cell():
    inputs = keras.Input(shape=(None, ModelConfig.L_FRAME // 2 + 1))

    rnn_cells = [layers.GRUCell(ModelConfig.HID_SIZE) for _ in range(ModelConfig.NUM_LAYERS)]
    stacked_gru = layers.StackedRNNCells(rnn_cells)
    output_rnn = layers.RNN(stacked_gru)(inputs)
    input_size = np.shape(inputs)[2]
    src1 = layers.Dense(input_size, activation="relu")(output_rnn)
    # src2_pre = layers.Dense(input_size, activation="relu")(output_rnn)

    # time-freq masking layer
    # src1 = src1_pre / (src1_pre + src2_pre + np.finfo(float).eps) * inputs
    # src2 = src2_pre / (src1_pre + src2_pre + np.finfo(float).eps) * inputs

    model = keras.Model(inputs=inputs, outputs=src1, name="GRUCell_model")
    model.summary()
    return model


def build_model_gru():
    inputs = keras.Input(shape=(None, ModelConfig.L_FRAME // 2 + 1))

    x = layers.GRU(ModelConfig.HID_SIZE, return_sequences=True)(inputs)
    x = layers.GRU(ModelConfig.HID_SIZE, return_sequences=True)(x)
    x = layers.GRU(ModelConfig.HID_SIZE, return_sequences=True)(x)
    input_size = np.shape(inputs)[2]
    src1_pre = layers.Dense(input_size, activation="relu")(x)
    src2_pre = layers.Dense(input_size, activation="relu")(x)

    # time-freq masking layer
    src1 = MaskLayer(name="src1")([src1_pre, src2_pre, inputs])
    src2 = MaskLayer(name="src2")([src2_pre, src1_pre, inputs])

    model = keras.Model(
        inputs=inputs,
        outputs=[src1, src2],
        name="GRU_model"
    )
    model.summary()
    # keras.utils.plot_model(model, "model.png", show_shapes=True)
    return model
