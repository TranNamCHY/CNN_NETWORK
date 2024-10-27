import numpy as np
from Layer import Layer
class Flatten(Layer):
  # A standard fully-connected layer with softmax activation.
  def __init__(self, name):
     super().__init__(name)
  def forward(self, input):
    '''
    This perform both dense and softmax.
    Performs a forward pass of the softmax layer using the given input.
    Returns a 1d numpy array containing the respective probability values.
    - input can be any array with any dimensions.
    '''
    self.last_input_shape = input.shape
    input = input.flatten()
    return input

  def backprop(self, d_L_d_out,learn_rate):
      return d_L_d_out.reshape(self.last_input_shape)
