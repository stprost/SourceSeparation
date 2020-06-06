from keras.engine import Layer
import numpy as np

class MaskLayer(Layer):

    def __init__(self, units=1, **kwargs):
        super(MaskLayer, self).__init__(**kwargs)
        self.units = units

    def call(self, inputs):  # Defines the computation from inputs to outputs
        return inputs[0] / (inputs[0] + inputs[1] + np.finfo(float).eps) * inputs[2]
