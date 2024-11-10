import numpy as np
from Layer import Layer
class Dense(Layer):
  # A standard fully-connected layer with softmax activation.

  def __init__(self, input_len, num_neuron, name, need_update):
    # We divide by input_len to reduce the variance of our initial values
    super().__init__(name)
    self.weights = np.random.randn(input_len, num_neuron).astype(np.float32) / input_len
    self.biases = np.zeros(num_neuron)
    self.need_update = need_update
  def forward(self, input):
    '''
    This perform both dense and softmax.
    Performs a forward pass of the softmax layer using the given input.
    Returns a 1d numpy array containing the respective probability values.
    - input can be any array with any dimensions.
    '''
    self.last_input_shape = input.shape

    #input = input.flatten()
    self.last_input = input

    #input_len, nodes = self.weights.shape

    '''
        Can replace with C mutiply function.
    '''
    #print(input.shape)
    #print(self.weights.shape)
    totals = np.dot(input, self.weights) + self.biases
    self.last_totals = totals

    #exp = np.exp(totals)
    #return exp / np.sum(exp, axis=0)
    return totals
  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the softmax layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    '''
      Explain: Assume the input is [1xN], the weight matrix is [NxM], the output is [1xM].
      - The input gradient d_L_d_out should be [1xM].
      - The gradient of Loss func against inputs should be [1xN].
    '''
    d_out_d_w = self.last_input
    d_out_d_b = 1
    d_out_d_inputs = self.weights
    d_L_d_w = d_out_d_w[np.newaxis].T @ d_L_d_out[np.newaxis]
    d_L_d_b = d_L_d_out * d_out_d_b
    d_L_d_inputs = d_L_d_out @ d_out_d_inputs.T
    #Update the new kernel and biases
    if self.need_update == True:
      self.weights -= learn_rate * d_L_d_w
      self.biases -= learn_rate * d_L_d_b

    #Return the gradient of loss function against the input of this layer aka the output of previous layer.
    return d_L_d_inputs
  def save_weight(self):
    np.save(self.name +  "_weight", self.weights)
    np.save(self.name +  "_bias", self.biases)
  def load_weight_by_name(self):
    self.weights = np.load(self.name +  "_weight.npy")
    self.biases = np.load(self.name +  "_bias.npy")