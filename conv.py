import numpy as np
import ctypes
from numpy.lib.stride_tricks import as_strided
from Layer import Layer
import time
import signal
import os
import fcntl
import time
import mmap
'''
   
'''
SET_PID_COMMAND = 0x40046401
PRE_SRC_BUFF = 0x40046402
PRE_KERNEL_BUFF = 0x40046403
PRE_DEST_BUFF = 0x4046404
SET_IMAGE_HEIGHT_WIDTH = 0x40046405
START_CACULATE = 0x40046406
SET_SIGNAL_NUMBER = 0x400464067
MAX_DEST_BUFFER = 100*100
MAX_SRC_BUFFER = 100*100
KERNEL_LEN = 9
SIG_TEST = 44
'''
Note: In this implementation, we assume the input is a 2d numpy array for simplicity, because that's
how our MNIST images are stored. This works for us because we use it as the first layer in our
network, but most CNNs have many more Conv layers. If we were building a bigger network that needed
to use Conv3x3 multiple times, we'd have to make the input be a 3d numpy array.
'''

class Conv3x3(Layer):
  # A Convolution layer using 3x3 filters.

  def __init__(self, num_filters, num_chan, name, fd=-1,src_buffer=None, dest_buffer=None, kernel_buffer=None,num_signal=-1):
    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    super().__init__(name)
    self.filters = np.random.randn(num_filters, 3, 3, num_chan).astype(np.float32) / 9
    self.num_filters = num_filters
    self.output_num_chan = num_chan
    self.rotated_filters = np.random.randn(num_filters, 3, 3, num_chan).astype(np.float32)
    self.rotated_filters[:,:,:,:]  = 0
    self.wait_flag = False

  def __init__(self, num_filters, num_chan, name, fd, src_buffer, dest_buffer, kernel_buffer, num_signal, type_conv, need_caculate_backprop, need_update_weight):
    super().__init__(name)
    self.filters = np.random.randn(num_filters, 3, 3, num_chan).astype(np.float32) / 9
    self.num_filters = num_filters
    self.output_num_chan = num_chan
    self.rotated_filters = np.random.randn(num_filters, 3, 3, num_chan).astype(np.float32)
    self.rotated_filters[:,:,:,:]  = 0
    self.wait_flag = False
    self.fd = fd
    self.num_signal = num_signal
    self.register_signal_handle()
    self.src_buffer = src_buffer
    self.dest_buffer = dest_buffer
    self.kernel_buffer = kernel_buffer
    self.type_conv = type_conv
    self.need_caculate_backprop = need_caculate_backprop
    self.need_update_weight = need_update_weight
    self.forward_time = 0
    self.backward_time = 0
    if self.type_conv == "fpga_forward":
      self.used_convo_op = self.fpga_backprop_custom_conv2d_8bit
    if self.type_conv == "int8bit_forward":
      self.used_convo_op = self.test_custom_conv2d_8bit
    if self.type_conv == "float32_forward":
      self.used_convo_op = self.test_custom_conv2d
    if self.type_conv == "test_fpga_forward":
      self.used_convo_op = self.test_fpga_backprop_custom_conv2d_8bit
    if self.type_conv == "new_conv":
      self.used_convo_op  = self.new_conv_op
  def free_resource(self):
     if self.type_conv == "fpga_forward":
      del self.src_matrix
      del self.dest_matrix
      del self.kernel_matrix
      del self.backprop_src_matrix
      del self.backprop_dest_matrix
  def signal_handler(self, sigunum,frame):
    self.wait_flag = True

  def register_signal_handle(self):
    signal.signal(self.num_signal, self.signal_handler)

  def set_image_size(self, height, width):
    fcntl.ioctl(self.fd, SET_IMAGE_HEIGHT_WIDTH, (height << 16) | width)

  def reshape_mmap_buffer(self, input_image_heitht, input_image_width):
    self.src_matrix = (np.frombuffer(self.src_buffer, dtype = np.int8))[:(input_image_heitht * input_image_width)].reshape(input_image_heitht,input_image_width)
    self.dest_matrix = np.frombuffer(self.dest_buffer, dtype=np.int32)[:((input_image_heitht - 2) * (input_image_width -2))].reshape(input_image_heitht - 2,input_image_width - 2)
    self.kernel_matrix = (np.frombuffer(self.kernel_buffer, dtype=np.int8)).reshape(3,3)
    self.backprop_src_matrix = (np.frombuffer(self.src_buffer, dtype = np.int8))[:((input_image_heitht+2) * (input_image_width+2))].reshape(input_image_heitht+2,input_image_width+2)
    self.backprop_dest_matrix = (np.frombuffer(self.dest_buffer, dtype = np.int32))[:(input_image_heitht * input_image_width)].reshape(input_image_heitht,input_image_width)
  def my_covolution_op(self, image, kernel):
    #first_mark_time = time.time()
    self.wait_flag = False
    self.src_matrix[:, :] = image[:, :].astype(np.int8)
    self.kernel_matrix[:, :] = kernel[:, :].astype(np.int8)
    #prepare_time = time.time() - first_mark_time

    #mark_time = time.time()
    fcntl.ioctl(self.fd, START_CACULATE, self.num_signal)
    #ioctl_time = time.time() - mark_time
    #sum_time = time.time() - first_mark_time
    #print([prepare_time, ioctl_time, sum_time])
    #print("Prepare time in con op: ", time.time() - first_mark_time)
    #mark_time = time.time()
    #while(1):
    #  if self.wait_flag == True:
    #    break
    #print("Time wait for interrupt: ", time.time() - mark_time)
    return self.dest_matrix.astype(np.float32)

  def my_back_prop_covolution_op(self, image, kernel):
    #first_mark_time = time.time()
    self.wait_flag = False
    self.backprop_src_matrix[:, :] = image[:, :].astype(np.int8)
    self.kernel_matrix[:, :] = kernel[:, :].astype(np.int8)
    #prepare_time = time.time() - first_mark_time

    #mark_time = time.time()
    fcntl.ioctl(self.fd, START_CACULATE, self.num_signal)
    #ioctl_time = time.time() - mark_time
    #sum_time = time.time() - first_mark_time
    #print([prepare_time, ioctl_time, sum_time])
    #print("Prepare time in con op: ", time.time() - first_mark_time)
    #mark_time = time.time()
    #while(1):
    #  if self.wait_flag == True:
    #    break
    #print("Time wait for interrupt: ", time.time() - mark_time)
    return self.backprop_dest_matrix.astype(np.float32)
  

  def iterate_regions(self, image):
    '''
      Input image has shape(height, width, channel.)
    '''
    h, w, num_chan = image.shape

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image[i:(i + 3), j:(j + 3),:]
        yield im_region, i, j
  def test_iterate_regions(self, image, num_filter):
    '''
      Input image has shape(height, width, channel.)
    '''
    h, w, num_chan = image.shape
    tempt_image = np.zeros(image.shape, dtype = np.float32)
    tempt_image = image
    expanded_tempt_image = np.expand_dims(tempt_image, axis=3)
    expanded_tempt_image = np.tile(expanded_tempt_image, (1, 1, 1, num_filter))
    for i in range(h - 2):
      for j in range(w - 2):
        im_region = expanded_tempt_image[i:(i + 3), j:(j + 3), :, :]
        check_im_region = image[i:(i + 3), j:(j + 3), :]
        yield im_region, check_im_region,i, j
  def test_custom_conv2d(self,input_tensor_image,conv2d_weight):
    lib = ctypes.CDLL('./libmatrix.so')
    image_height = len(input_tensor_image[:,0,0])
    image_width = len(input_tensor_image[0,:,0])
    input_image_num_channel = len(input_tensor_image[0,0,:])
    #print(image_height, image_width, input_image_num_channel)
    output_image_num_channel = len(conv2d_weight[:,0,0,0])
    if(input_image_num_channel != len(conv2d_weight[0,0,0,:])):
      print("Got error input image channel not equal input weight channel")
      return None
    #test_image[0,0:image_height,0:image_width,0]
    tempt_result = np.zeros((image_height - 2)*(image_width - 2), dtype = np.float32)
    result_accumulator = np.zeros((image_height - 2)*(image_width - 2), dtype = np.float32)
    final_result = np.zeros((image_height - 2, image_width - 2,output_image_num_channel), dtype = np.float32)
    #print([image_height,image_width,input_image_num_channel,output_image_num_channel])
    activition = "relu"
    tempt_sum = 0
    first_mark_time = time.time()
    for num_filter in range(0, output_image_num_channel):
        result_accumulator[:] = 0
        for num_channel in range(0, input_image_num_channel):
            mark_time = time.time()
            lib.convolution_2d(input_tensor_image[:, :,num_channel].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)), image_height,image_width,
                         conv2d_weight[num_filter,:,:,num_channel].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         tempt_result[:].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         activition.encode('utf-8'),
                         0)
            tempt_sum += (time.time() - mark_time)
            result_accumulator[:] = result_accumulator[:] + tempt_result[:]
        final_result[:,:,num_filter] = result_accumulator.reshape(image_height - 2,image_width - 2)
    #print("Time of convolution_2d C take: ",tempt_sum)
    return final_result
  
  def custom_conv2d(self,input_tensor_image,filter):
    lib = ctypes.CDLL('./libmatrix.so')
    image_height = len(input_tensor_image[:,0])
    image_width = len(input_tensor_image[0,:])
    #input_image_num_channel = len(input_tensor_image[0,0,:])
    #print(image_height, image_width, input_image_num_channel)
    #test_image[0,0:image_height,0:image_width,0]
    tempt_result = np.zeros((image_height - 2)*(image_width - 2), dtype = np.float32)
    #print([image_height,image_width,input_image_num_channel,output_image_num_channel])
    activition = "relu"
    ''' lib.convolution_2d(input_tensor_image[:,:].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)), image_height,image_width,
                         filter[:,:].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         tempt_result[:].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         activition.encode('utf-8'),
                         0) '''
    lib.convolution_2d(input_tensor_image[:,:].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)), image_height,image_width,
                         filter[:,:].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         tempt_result[:].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         activition.encode('utf-8'),
                         0)
    return tempt_result.reshape(image_height - 2,image_width - 2)

  def test_custom_conv2d_8bit(self,input_tensor_image,conv2d_weight):
    lib = ctypes.CDLL('./libmatrix.so')
    #test_lib = ctypes.CDLL('./libexample.so')
    '''
      Find the shape.
    '''
    #tempt_conv2d_weight = np.zeros(conv2d_weight.shape).astype(np.float32)
    #tempt_conv2d_weight[:,:,:,:] = conv2d_weight[:,:,:,:]
    tempt_conv2d_weight = conv2d_weight
    #tempt_input_tensor_image = np.zeros(input_tensor_image.shape).astype(np.float32)
    #tempt_input_tensor_image[:,:,:] = input_tensor_image[:,:,:]
    tempt_input_tensor_image = input_tensor_image
    image_height = len(input_tensor_image[:,0,0])
    image_width = len(input_tensor_image[0,:,0])
    input_image_num_channel = len(input_tensor_image[0,0,:])
    output_image_num_channel = len(conv2d_weight[:,0,0,0])

    #tempt_input_tensor_image = np.zeros(input_tensor_image.shape, dtype = np.int8)
    #tempt_conv2d_weight = np.zeros(conv2d_weight.shape, dtype = np.int8)
    tempt_result = np.zeros((image_height - 2)*(image_width - 2), dtype = np.int32)
    result_accumulator = np.zeros((image_height - 2)*(image_width - 2), dtype = np.float32)
    final_result = np.zeros((image_height - 2, image_width - 2,output_image_num_channel), dtype = np.float32)

    # vector_abs_max_image has shape(1) for each channel of input image.
    vector_abs_max_image = np.max(np.abs(tempt_input_tensor_image), axis = (0,1))
    #vector_abs_max_image = np.where(vector_abs_max_image == 0, 0.00001, vector_abs_max_image)
    # vector_abs_max_fitler has shape(1,1) for each channel of each filter.
    matrix_abs_max_fitler = np.max(np.abs(tempt_conv2d_weight), axis = (1,2))
    #matrix_abs_max_fitler = np.where(matrix_abs_max_fitler == 0, 0.0001, matrix_abs_max_fitler)
    #print(image_height, image_width, input_image_num_channel)
    '''
      Enusure that no divide by zeros exception will occur
    '''
    '''
      Revalue the input tensor image and kernel to range[-1,1].
    '''
    #input_tensor_image = input_tensor_image / vector_abs_max_image
    #conv2d_weight = conv2d_weight / matrix_abs_max_fitler[:, np.newaxis, np.newaxis, :]
    '''
      Revalue and round the input tensor image and kernel to range[-100,100]. 
    '''
    tempt_input_tensor_image = (tempt_input_tensor_image / vector_abs_max_image) * 100
    tempt_input_tensor_image =  tempt_input_tensor_image.astype(np.int8)
    tempt_conv2d_weight = (tempt_conv2d_weight / matrix_abs_max_fitler[:, np.newaxis, np.newaxis, :]) * 100
    tempt_conv2d_weight = tempt_conv2d_weight.astype(np.int8)
    if(input_image_num_channel != len(conv2d_weight[0,0,0,:])):
      print("Got error input image channel not equal input weight channel")
      return None
    #test_image[0,0:image_height,0:image_width,0]
    #print([image_height,image_width,input_image_num_channel,output_image_num_channel])
    #print(tempt_input_tensor_image)
    #print(tempt_conv2d_weight)\
    #test_lib.print_int8_array(tempt_input_tensor_image[:,:,:].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),3)
    tempt_sum = 0
    first_mark_time = time.time()
    for num_filter in range(0, output_image_num_channel):
        result_accumulator[:] = 0
        for num_channel in range(0, input_image_num_channel):
            #print(tempt_input_tensor_image[:, :,num_channel].flatten())
            mark_time = time.time()
            lib.convolution_2d_8bit(tempt_input_tensor_image[:, :,num_channel].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), image_height,image_width,
                        tempt_conv2d_weight[num_filter,:,:,num_channel].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                        tempt_result[:].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            tempt_sum += (time.time() - mark_time)
            #print(tempt_result)
            #mark_time = time.time()
            result_accumulator[:] = result_accumulator[:] + (tempt_result[:].astype(np.float32)/(100*100))*(vector_abs_max_image[num_channel] * matrix_abs_max_fitler[num_filter, num_channel])
        final_result[:,:,num_filter] = result_accumulator.reshape(image_height - 2,image_width - 2)
    #print("Outside the for loop: ")
    #print(time.time() - first_mark_time)
    #print("Total sum of convolution_2d_8bit: ", tempt_sum)
    return final_result
  def fpga_custom_conv2d_8bit(self, input_tensor_image, tempt_conv2d_weight):
    # Tinh ma tran chuyen vi
    self.set_image_size(len(input_tensor_image[:,0,0]), len(input_tensor_image[0,:,0]))
    conv2d_weight = np.transpose(tempt_conv2d_weight, (0,2,1,3))
    image_height = len(input_tensor_image[:,0,0])
    image_width = len(input_tensor_image[0,:,0])
    input_image_num_channel = len(input_tensor_image[0,0,:])
    output_image_num_channel = len(conv2d_weight[:,0,0,0])
    if(input_image_num_channel != len(conv2d_weight[0,0,0,:])):
      print("Got error input image channel not equal input weight channel")
      print([input_tensor_image.shape, conv2d_weight.shape])
      return None
    #tempt_input_tensor_image = np.zeros(input_tensor_image.shape, dtype = np.int8)
    #tempt_conv2d_weight = np.zeros(conv2d_weight.shape, dtype = np.int8)
    #tempt_result = np.zeros((image_height - 2)*(image_width - 2), dtype = np.int32)
    result_accumulator = np.zeros((image_height - 2, image_width - 2), dtype = np.float32)
    final_result = np.zeros((image_height - 2, image_width - 2,output_image_num_channel), dtype = np.float32)

    # vector_abs_max_image has shape(1) for each channel of input image.
    vector_abs_max_image = np.max(np.abs(input_tensor_image), axis = (0,1))
    # vector_abs_max_fitler has shape(1,1) for each channel of each filter.
    matrix_abs_max_fitler = np.max(np.abs(conv2d_weight), axis = (1,2))
    #print(image_height, image_width, input_image_num_channel)
    '''
      Revalue the input tensor image and kernel to range[-1,1].
    '''
    #input_tensor_image = input_tensor_image / vector_abs_max_image
    #conv2d_weight = conv2d_weight / matrix_abs_max_fitler[:, np.newaxis, np.newaxis, :]
    '''
      Revalue and round the input tensor image and kernel to range[-100,100]. 
    '''
    input_tensor_image = (input_tensor_image / vector_abs_max_image) * 100
    input_tensor_image =  input_tensor_image.astype(np.int8)
    conv2d_weight = (conv2d_weight / matrix_abs_max_fitler[:, np.newaxis, np.newaxis, :]) * 100
    conv2d_weight = conv2d_weight.astype(np.int8)
    #test_image[0,0:image_height,0:image_width,0]
    #print([image_height,image_width,input_image_num_channel,output_image_num_channel])
    #print(tempt_input_tensor_image)
    #print(tempt_conv2d_weight)\
    #test_lib.print_int8_array(tempt_input_tensor_image[:,:,:].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),3)
    tempt_sum = 0
    for num_filter in range(0, output_image_num_channel):
        result_accumulator[:] = 0
        for num_channel in range(0, input_image_num_channel):
            #print(tempt_input_tensor_image[:, :,num_chfannel].flatten())
            ''' lib.convolution_2d_8bit(input_tensor_image[:, :,num_channel].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), image_height,image_width,
                        conv2d_weight[num_filter,:,:,num_channel].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                        tempt_result[:].ctypes.data_as(ctypes.POINTER(ctypes.c_int32))) '''
            #print(tempt_result)
            mark_time = time.time()
            result_accumulator[:, :] = result_accumulator[:, :] + ((self.my_covolution_op(image=input_tensor_image[:, : , num_channel], 
                                                                                          kernel=conv2d_weight[num_filter, :, :, num_channel]))/(100*100))*(vector_abs_max_image[num_channel] * matrix_abs_max_fitler[num_filter, num_channel])
            tempt_sum += (time.time() - mark_time)
        final_result[:,:,num_filter] = result_accumulator
    #print("Total sum for my_covolution_op: ", tempt_sum)
    return final_result
  
  def test_fpga_backprop_custom_conv2d_8bit(self, input_tensor_image, tempt_conv2d_weight):
    return self.fpga_backprop_custom_conv2d_8bit(input_tensor_image, np.transpose(tempt_conv2d_weight, (0,2,1,3)))
  def fpga_backprop_custom_conv2d_8bit(self, input_tensor_image, tempt_conv2d_weight):
    #Tinh ma tran chuyen vi

    image_height = len(input_tensor_image[:,0,0])
    image_width = len(input_tensor_image[0,:,0])
    self.set_image_size(image_height,image_width)
    conv2d_weight = np.transpose(tempt_conv2d_weight, (0,2,1,3))
    input_image_num_channel = len(input_tensor_image[0,0,:])
    output_image_num_channel = len(conv2d_weight[:,0,0,0])
    if(input_image_num_channel != len(conv2d_weight[0,0,0,:])):
      print("Got error input image channel not equal input weight channel")
      print([input_tensor_image.shape, conv2d_weight.shape])
      return None
    #tempt_input_tensor_image = np.zeros(input_tensor_image.shape, dtype = np.int8)
    #tempt_conv2d_weight = np.zeros(conv2d_weight.shape, dtype = np.int8)
    #tempt_result = np.zeros((image_height - 2)*(image_width - 2), dtype = np.int32)
    result_accumulator = np.zeros((image_height - 2, image_width - 2), dtype = np.float32)
    final_result = np.zeros((image_height - 2, image_width - 2,output_image_num_channel), dtype = np.float32)

    # vector_abs_max_image has shape(1) for each channel of input image.
    vector_abs_max_image = np.max(np.abs(input_tensor_image), axis = (0,1))
    # vector_abs_max_fitler has shape(1,1) for each channel of each filter.
    matrix_abs_max_fitler = np.max(np.abs(conv2d_weight), axis = (1,2))
    
    #print(image_height, image_width, input_image_num_channel)
    '''
      Revalue the input tensor image and kernel to range[-1,1].
    '''
    #input_tensor_image = input_tensor_image / vector_abs_max_image
    #conv2d_weight = conv2d_weight / matrix_abs_max_fitler[:, np.newaxis, np.newaxis, :]
    '''
      Revalue and round the input tensor image and kernel to range[-100,100]. 
    '''
    input_tensor_image = (input_tensor_image / vector_abs_max_image) * 100
    input_tensor_image =  input_tensor_image.astype(np.int8)
    conv2d_weight = (conv2d_weight / matrix_abs_max_fitler[:, np.newaxis, np.newaxis, :]) * 100
    conv2d_weight = conv2d_weight.astype(np.int8)
    #test_image[0,0:image_height,0:image_width,0]
    #print([image_height,image_width,input_image_num_channel,output_image_num_channel])
    #print(tempt_input_tensor_image)
    #print(tempt_conv2d_weight)\
    #test_lib.print_int8_array(tempt_input_tensor_image[:,:,:].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),3)
    tempt_sum = 0
    for num_filter in range(0, output_image_num_channel):
        result_accumulator[:] = 0
        for num_channel in range(0, input_image_num_channel):
            #print(tempt_input_tensor_image[:, :,num_chfannel].flatten())
            ''' lib.convolution_2d_8bit(input_tensor_image[:, :,num_channel].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), image_height,image_width,
                        conv2d_weight[num_filter,:,:,num_channel].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                        tempt_result[:].ctypes.data_as(ctypes.POINTER(ctypes.c_int32))) '''
            #print(tempt_result)
            mark_time = time.time()
            result_accumulator[:, :] = result_accumulator[:, :] + ((self.my_back_prop_covolution_op(image=input_tensor_image[:, : , num_channel], 
                                                                                          kernel=conv2d_weight[num_filter, :, :, num_channel]))/(100*100))*(vector_abs_max_image[num_channel] * matrix_abs_max_fitler[num_filter, num_channel])
            tempt_sum += (time.time() - mark_time)
        final_result[:,:,num_filter] = result_accumulator
    #print("Total sum for my_covolution_op: ", tempt_sum)
    return final_result

  def forward(self, input):
    mark_time = time.time()
    if self.type_conv == "fpga_forward":
      result = self.fpga_forward(input)
    if self.type_conv == "int8bit_forward":
      result = self.int8bit_forward(input)
    if self.type_conv == "float32_forward":
      result = self.float_forward(input)
    if self.type_conv  == "test_fpga_forward":
      result = self.test_fpga_forward(input)
    if self.type_conv ==  "new_conv":
      result = self.new_con_forward(input)
    self.forward_time += (time.time() - mark_time)
    return result
  def save_weight(self):
    np.save(self.name + "_weight",self.filters)
  def load_weight_by_name(self):
    self.filters = np.load(self.name + "_weight.npy")
  def float_forward(self, input):
    self.last_input = input
    ''' if not hasattr(self,'src_matrix'):
      self.reshape_mmap_buffer(len(input[:,0,0]), len(input[0,:,0]))
    self.set_image_size(len(input[:,0,0]), len(input[0,:,0]))
    return self.fpga_custom_conv2d_8bit(input_tensor_image=input, conv2d_weight=self.filters) '''
    return self.test_custom_conv2d(input, self.filters)
  def int8bit_forward(self, input):
    self.last_input = input
    #tempt = self.filters
    #print(tempt)
    #ptr = self.filters.ctypes.data
    #print(f"Address of self.filter : {ptr}")
    #test_filter = np.transpose(self.filters, (0,2,1,3))
    t = self.test_custom_conv2d_8bit(input, self.filters) 
    #ptr = self.filters.ctypes.data
    #print(f"Address of self.filter : {ptr}")
    #print("After: ",self.filters)
    #print("diffrence: ",np.amax(self.filters - tempt))
    return t
  def new_con_forward(self,input):
    self.last_input = input
    return self.new_conv_op(input, self.filters)
  def fpga_forward(self, input):
    self.last_input = input
    if not hasattr(self,'src_matrix'):
      self.reshape_mmap_buffer(len(input[:,0,0]), len(input[0,:,0]))
    #mark_time = time.time()
    #self.set_image_size(len(input[:,0,0]), len(input[0,:,0]))
    #print("Set image size take: ")
    #print(time.time() - mark_time)
    #print(self.filters.shape)
    return self.fpga_custom_conv2d_8bit(input_tensor_image=input,tempt_conv2d_weight=self.filters)
    #return self.fpga_custom_conv2d_8bit(input_tensor_image=input, conv2d_weight = self.filters)
  def test_fpga_forward(self,input):
    self.last_input = input
    if not hasattr(self,'src_matrix'):
      self.reshape_mmap_buffer(len(input[:,0,0]), len(input[0,:,0]))
    #mark_time = time.time()
    #self.set_imhttps://www.youtube.com/shorts/qC1cejwGFTMage_size(len(input[:,0,0]), len(input[0,:,0]))
    #print("Set image size take: ")
    #print(time.time() - mark_time)
    #print(self.filters.shape)
    return self.fpga_custom_conv2d_8bit(input_tensor_image=input,tempt_conv2d_weight=np.transpose(self.filters, (0,2,1,3)))
    #return self.fpga_custom_conv2d_8bit(input_tensor_image=input, conv2d_weight = self.filters)
  def old_forward(self, input):
    '''
      Filters has shape(num_filters, 3,3, channel).
      Input image has shape(height, width, channel).
    '''
    ''' self.last_input = input
    return self.custom_conv2d(input, .filters) '''
    '''self
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    self.last_input = input

    h, w = input.shape
    output = np.zeros((h - 2, w - 2, self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
    return output
  def caculate_backprop(self, padded_previous_loss, rotated_kernel):
    height, width, num_channel = padded_previous_loss.shape
    output = np.zeros((height - 2, width - 2,1))
    for im_region, i , j in self.iterate_regions(padded_previous_loss):
        output[i,j] += np.sum(im_region * rotated_kernel,axis=(0,1))
  def previous_error_at_Nth_channel(self, Nth, padded_d_L_d_out, tempt):
    tempt[:, :] = 0
    tempt_sum = 0
    for i in range(0, self.num_filters):
      #print(padded_d_L_d_out.shape)
      mark_time = time.time()
      tempt += self.custom_conv2d(padded_d_L_d_out[:,:,i], self.rotated_filters[i,:,:,Nth])
      #tempt += self.test(padded_d_L_d_out[:,:,i], self.rotated_filters[i,:,:,Nth])
      tempt_sum +=  (time.time() - mark_time)
    return tempt,tempt_sum
  def caculate_previos_error(self,d_L_d_out):
    mark_time = time.time()
    image_height, image_widh, image_num_chan = self.last_input.shape
    previous_error_matrix = np.zeros((image_height, image_widh, image_num_chan), dtype = np.float32)
    # Add padding for d_L_d_out
    padded_input_error = self.add_zeros_padding(d_L_d_out, self.num_filters)
    tempt = np.zeros((image_height, image_widh), dtype=np.float32)
    tempt_sum = 0
    for i in range(0, image_num_chan):
      previous_error_matrix[:,:,i],t  = self.previous_error_at_Nth_channel(Nth=i,padded_d_L_d_out=padded_input_error, tempt=tempt)
      tempt_sum += t
      #print(previous_error_matrix[:,:,i])
    #print(previous_error_matrix)
    print("caculate_previos_error take: ")
    print(time.time() - mark_time)
    #print("Actual conv2d take: ")
    #print(tempt_sum)
    return previous_error_matrix
  def test_caculate_previos_error(self,d_L_d_out):
    mark_time = time.time()
    image_height, image_widh, image_num_chan = self.last_input.shape
    previous_error_matrix = np.zeros((image_height, image_widh, image_num_chan), dtype = np.float32)
    # Add padding for d_L_d_out
    padded_input_error = self.add_zeros_padding(d_L_d_out, self.num_filters)
    tempt = np.zeros((image_height, image_widh), dtype=np.float32)
    tempt_sum = 0
    ''' for i in range(0, image_num_chan):
      previous_error_matrix[:,:,i],t  = self.previous_error_at_Nth_channel(Nth=i,padded_d_L_d_out=padded_input_error, tempt=tempt)
      tempt_sum += t '''
    previous_error_matrix = self.test_custom_conv2d(padded_input_error, np.transpose(self.rotated_filters,(3,1,2,0)))
    #print(previous_error_matrix)
    #print("test_caculate_previos_error take: ")
    #print(time.time() - mark_time)
    #print("Actual conv2d take: ")
    #print(tempt_sum)
    return previous_error_matrix
  def test_8bit_caculate_previos_error(self,d_L_d_out):
    mark_time = time.time()
    image_height, image_widh, image_num_chan = self.last_input.shape
    previous_error_matrix = np.zeros((image_height, image_widh, image_num_chan), dtype = np.float32)
    # Add padding for d_L_d_out
    padded_input_error = self.add_zeros_padding(d_L_d_out, self.num_filters)
    #tempt = np.zeros((image_height, image_widh), dtype=np.float32)
    #tempt_sum = 0
    ''' for i in range(0, image_num_chan):
      previous_error_matrix[:,:,i],t  = self.previous_error_at_Nth_channel(Nth=i,padded_d_L_d_out=padded_input_error, tempt=tempt)
      tempt_sum += t '''
    #print(padded_input_error.shape)
    #print(self.name)
    previous_error_matrix = self.used_convo_op(padded_input_error, np.transpose(self.rotated_filters,(3,1,2,0)))
    #print(previous_error_matrix)
    #print("test_8bit_caculate_previos_error take: ", time.time() - mark_time)
    #print(time.time() - mark_time)
    #print("Actual conv2d take: ")
    #print(tempt_sum)
    #print("caculate previous error at: " + self.name)
    #print(previous_error_matrix[:,:,0])
    return previous_error_matrix
  def test_fpga_caculate_previos_error(self,d_L_d_out):
    mark_time = time.time()
    image_height, image_widh, image_num_chan = self.last_input.shape
    previous_error_matrix = np.zeros((image_height, image_widh, image_num_chan), dtype = np.float32)
    # Add padding for d_L_d_out
    padded_input_error = self.add_zeros_padding(d_L_d_out, self.num_filters)
    tempt = np.zeros((image_height, image_widh), dtype=np.float32)
    tempt_sum = 0
    ''' for i in range(0, image_num_chan):
      previous_error_matrix[:,:,i],t  = self.previous_error_at_Nth_channel(Nth=i,padded_d_L_d_out=padded_input_error, tempt=tempt)
      tempt_sum += t '''
    previous_error_matrix = self.test_custom_conv2d_8bit(padded_input_error, np.transpose(self.rotated_filters,(3,1,2,0)).T)
    #print(previous_error_matrix)
    #print("test_8bit_caculate_previos_error take: ")
    #print(time.time() - mark_time)
    #print("Actual conv2d take: ")
    #print(tempt_sum)
    return previous_error_matrix
  def inner_forward(self, input, filters):
    tempt = self.forward(input, filters)
    return np.sum(tempt, axis=2)
  def rotate_180degree(self,filters, num_filter):
    for i in range(0,num_filter):
      self.rotated_filters[i] = np.rot90(filters[i],2)
    return self.rotated_filters
  def add_zeros_padding(self,filter, num_filter):
    return np.pad(filter, ((2, 2), (2, 2), (0, 0)), mode='constant')
 
  def sliding_window_view_4d_reverse(self,arr, window_shape):
    """
    Create a sliding window view of a 4D input array along the 1st and 2nd dimensions.
    
    Parameters:
        arr: The input 4D array (shape: (a, b, c, d)).
        window_shape: Tuple specifying the shape of the sliding window for the 1st and 2nd dimensions.
        
    Returns:
        A view of the array with sliding windows applied along the 1st and 2nd dimensions.
    """
    # Check that the input is 4D
    if arr.ndim != 4:
        raise ValueError("Input array must be 4-dimensional")

    # Get the shape of the input array
    a, b, c, d = arr.shape
    
    # Check the window shape
    if len(window_shape) != 2:
        raise ValueError("Window shape must have two dimensions (for the 1st and 2nd dimensions of the array)")

    # Define the output shape: 
    # For the 1st and 2nd dimensions, reduce based on window_shape
    # Keep the 3rd and 4th dimensions the same
    out_shape = (a - window_shape[0] + 1, b - window_shape[1] + 1, window_shape[0], window_shape[1], c, d)

    # Define the strides: 
    # Strides for the 1st and 2nd dimensions are modified to enable sliding windows
    # Strides for the 3rd and 4th dimensions remain the same
    strides = arr.strides[:2] + arr.strides[:2] + arr.strides[2:]

    # Return the sliding window view using as_strided
    return as_strided(arr, shape=out_shape, strides=strides)

  def sliding_window_view_3d_reverse(self,arr, window_shape):
    """
    Create a sliding window view of a 4D input array along the 1st and 2nd dimensions.
    
    Parameters:
        arr: The input 4D array (shape: (a, b, c, d)).
        window_shape: Tuple specifying the shape of the sliding window for the 1st and 2nd dimensions.
        
    Returns:
        A view of the array with sliding windows applied along the 1st and 2nd dimensions.
    """
    # Check that the input is 4D
    if arr.ndim != 3:
        raise ValueError("Input array must be 3-dimensional")

    # Get the shape of the input array
    a, b, c = arr.shape
    
    # Check the window shape
    if len(window_shape) != 2:
        raise ValueError("Window shape must have two dimensions (for the 1st and 2nd dimensions of the array)")

    # Define the output shape: 
    # For the 1st and 2nd dimensions, reduce based on window_shape
    # Keep the 3rd and 4th dimensions the same
    out_shape = (a - window_shape[0] + 1, b - window_shape[1] + 1, window_shape[0], window_shape[1], c)

    # Define the strides: 
    # Strides for the 1st and 2nd dimensions are modified to enable sliding windows
    # Strides for the 3rd and 4th dimensions remain the same
    strides = arr.strides[:2] + arr.strides[:2] + arr.strides[2:]

    # Return the sliding window view using as_strided
    return as_strided(arr, shape=out_shape, strides=strides)
    
  def backprop(self, d_L_d_out, learn_rate):
    mark_time = time.time()
    if self.need_caculate_backprop == False:
      return None
    d_L_d_filters = np.zeros(self.filters.shape)
    tempt_image = np.zeros(self.last_input.shape, dtype = np.float32)
    tempt_image = self.last_input
    expanded_tempt_image = np.expand_dims(tempt_image, axis=3)
    expanded_tempt_image = np.tile(expanded_tempt_image, (1, 1, 1, self.num_filters))
    windows = self.sliding_window_view_4d_reverse(expanded_tempt_image,(3,3))
    tempt = windows * d_L_d_out[:, :, np.newaxis, np.newaxis, np.newaxis, :]
    tempt = np.sum(tempt, axis=(0,1))
    d_L_d_filters = np.transpose(tempt, (3,0,1,2))

    '''height = len(d_L_d_out[:,0,0])
    width = len(d_L_d_out[0,:,0])
    windows_image = self.sliding_window_view_3d_reverse(self.last_input,(height,width))
    result = windows_image[:,:,:,:,np.newaxis,:] * d_L_d_out[np.newaxis,np.newaxis,:,:,:,np.newaxis] 
    result = np.sum(result, axis = (2,3))
    d_L_d_filters = np.transpose(result, (2,0,1,3))'''

    ''''
      Update filters.
    '''
    if self.need_update_weight == True:
      self.filters[:,:,:,:] -= learn_rate * d_L_d_filters[:,:,:,:]
    self.rotate_180degree(self.filters, self.num_filters)
    result = self.test_8bit_caculate_previos_error(d_L_d_out)
    self.backward_time += (time.time() - mark_time)
    return result
  def new_conv_op(self,input,filters):
    windows_image = self.sliding_window_view_3d_reverse(input,(3,3))
    tempt = np.transpose(filters,(1,2,3,0))
    result = windows_image[:,:,:,:,:,np.newaxis]*tempt[np.newaxis,np.newaxis,:,:,:,:]
    result = np.sum(result, axis = (2,3,4))
    return result
'''
  Test
'''

''' conv = Conv3x3(num_filters=8,num_chan=1) 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images[0:1000].astype(np.float32)
print(conv.forward(train_images[0]))
print("Test conv: ")
print(conv.test_forward(train_images[0])) '''

''' test_image = np.random.randn(4,4,5).astype(np.float32)
test_filters = np.random.randn(2,2,16).astype(np.float32)
conv = Conv3x3(num_filters=8,num_chan=8, name="First_Conv",type_conv="int8bit_forward"
               , fd=None,src_buffer=None,dest_buffer=None,kernel_buffer=None,num_signal=SIG_TEST, need_caculate_backprop=False, need_update_weight=True)  
t = conv.test_new_conv_op(test_image, test_filters)
print(t.shape) '''