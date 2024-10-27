import numpy as np
from Layer import Layer
class tempt_Softmax(Layer):
  # A standard fully-connected layer with softmax activation
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
    self.last_input = input.astype(np.float64)
    try:
      # Example of an overflow (e.g. large exponential)
      exp = np.exp(self.last_input)
    except RuntimeWarning as e:
      print(f"RuntimeWarning caught: {e}")
      print("Value of input: ",input)
    # You can handle it here, such as assigning a large value or setting result to np.inf
      result = np.inf
    sum = np.sum(exp, axis=0)
    self.tempt_exp = exp
    self.tempt_sum = sum
    return exp / sum

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the softmax layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    # We know only 1 element of d_L_d_out will be nonzero
    for i, gradient in enumerate(d_L_d_out):
      if gradient == 0:
        continue

      # e^totals
      tempt_exp = self.tempt_exp

      # Sum of all e^totals
      tempt_sum = self.tempt_sum

      # Gradients of out[i] against totals
      # This is the gradient of output of softmax against the input of softmax.
      try:
        d_out_d_input = -tempt_exp[i] * tempt_exp / (tempt_sum ** 2)
        d_out_d_input[i] = tempt_exp[i] * (tempt_sum - tempt_exp[i]) / (tempt_sum ** 2)
      except RuntimeWarning as e:
        print(f"RuntimeWarning caught: {e}")
        print("Value of tempt_sum: ",tempt_sum)
      # Gradients of totals against weights/biases/input
      # Gradient of output of dense against  weight/biases/input.
      ''' d_t_d_w = self.last_input
      d_t_d_b = 1
      d_t_d_inputs = self.weights '''

      # Gradients of loss against totals
      # Gradients of loss func against output of dense layer
      # gradient variable aka the gradient of loss func against the output of softmax
      d_L_d_input = gradient * d_out_d_input

      # Gradients of loss against weights/biases/input
      # Gradients of loss func against weight/biases/input
      ''' d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
      d_L_d_b = d_L_d_t * d_t_d_b
      d_L_d_inputs = d_t_d_inputs @ d_L_d_t  

      # Update weights / biases
      self.weights -= learn_rate * d_L_d_w
      self.biases -= learn_rate * d_L_d_b '''

      return d_L_d_input.astype(np.float32)
