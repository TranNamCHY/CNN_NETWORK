import numpy as np
from Layer import Layer
import ctypes
import time
class MaxPool2(Layer):
  # A Max Pooling layer using a pool size of 2.

  def __init__(self, name, type_maxpool):
    super().__init__(name)
    self.type_maxpool = type_maxpool
    if self.type_maxpool == "float_forward":
      self.backprop = self.float_backprop
    if self.type_maxpool == "fpga_forward":
      self.backprop = self.fpga_backprop
    self.forward_time = 0
  def iterate_regions(self, image):
    '''
    Generates non-overlapping 2x2 image regions to pool over.
    - image is a 2d numpy array
    '''
    h, w, _ = image.shape
    new_h = h // 2
    new_w = w // 2

    for i in range(new_h):
      for j in range(new_w):
        im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
        yield im_region, i, j
  def find_max_coordinate(self, im_region):
    max_indices_flat = np.argmax((im_region.reshape(-1, im_region.shape[2])), axis=0)
    max_coords = np.unravel_index(max_indices_flat, (im_region.shape[0], im_region.shape[1]))
    return max_coords
  
  def forward(self, input): 
    mark_time = time.time()
    if self.type_maxpool == "float_forward":
      result = self.float_forward(input)
    if self.type_maxpool == "fpga_forward":
      result = self.fpga_forward(input)
    self.forward_time += (time.time() - mark_time)
    return result
  def float_forward(self, input):
    '''
    Performs a forward pass of the maxpool layer using the given input.
    Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
    - input is a 3d numpy array with dimensions (h, w, num_filters)
    '''
    self.last_input = input

    h, w, num_filters = input.shape
    output = np.zeros((h // 2, w // 2, num_filters), dtype=np.float32)
    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.amax(im_region, axis=(0, 1))
    return output
  
  def maxpool2d_multi_channel(self, image, pool_size=(2, 2)):
    # Get input image dimensions: (height, width, channels)
    self.last_input = image
    h, w, c = image.shape
    
    # Pool size dimensions (height and width of pooling window)
    pool_h, pool_w = pool_size
    
    # Ensure the height and width are divisible by the pool size (no padding here)
    h_out = h // pool_h
    w_out = w // pool_w
    
    # Reshape the image to get pooling windows across the height and width dimensions
    # The new shape will be (h_out, pool_h, w_out, pool_w, channels)
    reshaped = image[:h_out * pool_h, :w_out * pool_w, :].reshape(h_out, pool_h, w_out, pool_w, c)
    
    # Perform max pooling by taking the maximum over the pooling windows (axis 1 and 3 for height and width)
    pooled = reshaped.max(axis=(1, 3))  # This keeps the channel dimension intact
    
    return pooled
  def maxpool2d_with_indices(self, image, pool_size=(2, 2)):
    # Get input image dimensions (height, width, channels)
    self.last_input = image
    h, w, c = image.shape
    
    # Pool size dimensions (height and width of pooling window)
    pool_h, pool_w = pool_size
    
    # Ensure the height and width are divisible by the pool size (no padding here)
    h_out = h // pool_h
    w_out = w // pool_w
    
    # Reshape the image into (h_out, pool_h, w_out, pool_w, channels)
    reshaped = image[:h_out * pool_h, :w_out * pool_w, :].reshape(h_out, pool_h, w_out, pool_w, c)
    
    # Perform max pooling by taking the maximum over the height and width of the pool
    pooled = reshaped.max(axis=(1, 3))  # (h_out, w_out, channels)
    
    # Get the indices of the maximum values along axis (1, 3) (local to the pooling window)
    # First, flatten the pooling windows (pool_h * pool_w) so we can find the argmax per window
    reshaped_flat = reshaped.reshape(h_out, w_out, pool_h * pool_w, c)  # Shape (h_out, w_out, pool_h * pool_w, channels)
    max_indices = reshaped_flat.argmax(axis=2)  # Shape (h_out, w_out, channels)
    
    # Convert flattened indices to 2D indices (relative to the pool window)
    max_indices_h = max_indices // pool_w  # Row (height) index within the pooling window
    max_indices_w = max_indices % pool_w   # Column (width) index within the pooling window
    
    # Now, convert the local indices to global coordinates in the original imagef
    global_y_coords = np.arange(h_out)[:, None] * pool_h + max_indices_h  # Global y coordinates
    global_x_coords = np.arange(w_out)[None, :] * pool_w + max_indices_w  # Global x coordinates
    
    # Stack the global coordinates into a (h_out, w_out, channels, 2) array (y, x coordinates)
    final_coords = np.stack((global_y_coords[..., None], global_x_coords[..., None]), axis=-1)
    
    return pooled, final_coords
  
  def fpga_forward(self, input):
    '''
    Performs a forward pass of the maxpool layer using the given input.
    Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
    - input is a 3d numpy array with dimensions (h, w, num_filters)
    '''
    self.last_input = input
    output, self.x_coordinate_matrix, self.y_coordinate_matrix = self.custom_maxpool_layer(input,2)
    return output
  
  
  def custom_maxpool_layer(self, input_tensor_image, pool_size):
    lib = ctypes.CDLL('./libmatrix.so')
    image_height = len(input_tensor_image[:,0,0])
    image_width = len(input_tensor_image[0,:,0])
    input_image_num_channel = len(input_tensor_image[0,0,:])
    x_coordinate_matrix = np.zeros(int(image_height/pool_size) * int(image_width/pool_size) * int(input_image_num_channel), dtype = np.int32)
    y_coordinate_matrix = np.zeros(int(image_height/pool_size) * int(image_width/pool_size) * int(input_image_num_channel), dtype = np.int32)
    tempt_result = np.zeros(int(image_height/pool_size) * int(image_width/pool_size) * int(input_image_num_channel), dtype = np.float32)
    #print("In custom mapool layer")
    #print([int(image_height/pool_size), image_width, input_image_num_channel])
    lib.maxpool_2d(input_tensor_image[:,:,:].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),image_height,image_width,input_image_num_channel,
                #final_result[0,:,:,num_chan].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                tempt_result[:].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                , pool_size,
                x_coordinate_matrix[:].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                y_coordinate_matrix[:].ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        #final_result[:,:,num_chan] = tempt_result[:].reshape(int(image_height/pool_size), int(image_width/pool_size))
    return tempt_result.reshape(int(image_height/pool_size),int(image_width/pool_size),int(input_image_num_channel)), x_coordinate_matrix.reshape(int(image_height/pool_size),int(image_width/pool_size),int(input_image_num_channel)), y_coordinate_matrix.reshape(int(image_height/pool_size),int(image_width/pool_size),int(input_image_num_channel))

  def float_backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the maxpool layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    '''
    d_L_d_input = np.zeros((self.last_input.shape), dtype = np.float32)

    for im_region, i, j in self.iterate_regions(self.last_input):
      h, w, f = im_region.shape
      amax = np.amax(im_region, axis=(0, 1))

      for i2 in range(h):
        for j2 in range(w):
          for f2 in range(f):
            # If this pixel was the max value, copy the gradient to it.
            if im_region[i2, j2, f2] == amax[f2]:
              d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

    return d_L_d_input
  
  def fpga_backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the maxpool layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    '''
    d_L_d_input = np.zeros((self.last_input.shape), dtype = np.float32)
    h,w,num_chan = self.last_input.shape
    channel_indices = np.arange(num_chan).reshape(1, 1, num_chan)
    d_L_d_input[self.x_coordinate_matrix, self.y_coordinate_matrix, channel_indices] = d_L_d_out[:,:,:]
    return d_L_d_input