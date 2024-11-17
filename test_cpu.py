import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
from tempt_softmax import tempt_Softmax
from flatten import Flatten
from dense import Dense
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import ctypes
import numpy as np
import os
import signal
import time
import fcntl
import struct
import mmap
import timeit
import warnings
import functools
SET_PID_COMMAND = 0x40046401
PRE_SRC_BUFF = 0x40046402
PRE_KERNEL_BUFF = 0x40046403
PRE_DEST_BUFF = 0x40046404
SET_IMAGE_HEIGHT_WIDTH = 0x40046405
START_CACULATE = 0x40046406
FORCE_START_CACULATE = 0x40046407
MAX_DEST_BUFFER = 4*80*80
MAX_SRC_BUFFER = 100*100
KERNEL_LEN = 9
SIG_TEST = 44

def set_pid(fd):
    pid = os.getpid()
    fcntl.ioctl(fd, SET_PID_COMMAND, pid)

def prepare_mmap_buffer(fd):
    fcntl.ioctl(fd, PRE_SRC_BUFF, 0)
    src_buffer = mmap.mmap(fd, length=MAX_SRC_BUFFER, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ|mmap.PROT_WRITE, offset=0, access=mmap.ACCESS_WRITE)

    fcntl.ioctl(fd, PRE_DEST_BUFF, 0)
    dest_buffer = mmap.mmap(fd, length=MAX_DEST_BUFFER, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ|mmap.PROT_WRITE, offset=0, access=mmap.ACCESS_WRITE) 

    fcntl.ioctl(fd, PRE_KERNEL_BUFF, 0)
    kernel_buffer = mmap.mmap(fd, length=KERNEL_LEN, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ|mmap.PROT_WRITE, offset=0, access=mmap.ACCESS_WRITE)
    
    return src_buffer, dest_buffer, kernel_buffer

face_train_image = np.load('train_image.npy')
face_train_label = np.load('train_label.npy')
train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
face_test_images = np.load('test_image.npy')
face_test_labels = np.load('test_label.npy')
idex_face_train_label = np.load('idex_train_label.npy')
new_face_train_image = np.load('new_face_train_image.npy')
new_idex_face_train_label = np.load('new_idex_face_train_label.npy')
train_images = train_images[0:1000].astype(np.float32)
train_images = np.expand_dims(train_images, axis = -1)
train_labels = train_labels[0:1000]
test_image = np.zeros((28,28,1), dtype=np.float32)
test_image = train_images[0].astype(np.float32)

Sequential = []

conv = Conv3x3(num_filters=16,num_chan=3, name="First_Conv",type_conv="new_conv"
               , fd=None,src_buffer=None,dest_buffer=None,kernel_buffer=None,num_signal=SIG_TEST, need_caculate_backprop=False, need_update_weight=True)                  # 28x28x1 -> 26x26x16
Sequential.append(conv)                                                                                                                                      # 64x64x3 -> 62x62x16

pool = MaxPool2(name="First Maxpool", type_maxpool="fpga_forward")                  # 26x26x16 -> 13x13x16                                       
Sequential.append(pool)                                # 62x62x16 -> 31x31x16

second_conv = Conv3x3(num_filters=16,num_chan=16,name="Second_Conv",type_conv="new_conv"
                      , fd=None,src_buffer=None,dest_buffer=None,kernel_buffer=None,num_signal=(SIG_TEST+1), need_caculate_backprop=True, need_update_weight=True) # 13x13x16 -> 11x11x16
Sequential.append(second_conv)                                                                                                                     # 31x31x16 -> 29x29x16                                                                       

second_pool = MaxPool2(name="Second Maxpool", type_maxpool="fpga_forward")  # 11x11x16 -> 5x5x16
Sequential.append(second_pool)                 # 29x29x16 -> 14x14x16

flatten = Flatten(name="Flatten") # 5x5x16 -> 13*13*32
Sequential.append(flatten)        

dense1 = Dense(input_len=14*14*16, num_neuron=16, name="Dense1", need_update = True) # 5*5*16 -> 64
Sequential.append(dense1)

t_softmax = tempt_Softmax(name="Softmax") # 10 -> 10
Sequential.append(t_softmax)

def forward(image, label, Sequential): 
  out = image/255 - 0.5
  for layer in Sequential:
    out = layer.forward(out)
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0
  return out, loss, acc

def predict(image, Sequential):
  out = image/255 - 0.5
  for layer in Sequential:
    out = layer.forward(out)
  return out

def train(im, label, Sequential, Output_neuron, lr=.005):
  out, loss, acc = forward(im, label,Sequential)
  gradient = np.zeros(Output_neuron)
  gradient[label] = -1 / out[label]
  for layer in reversed(Sequential):
    gradient = layer.backprop(gradient, lr)
  return loss, acc

def fit(Sequential,train_images, train_labels, epoch):
  accur_each_epoch  = 0
  for idex_epoch in range(epoch):
    print('--- Epoch %d ---' % (idex_epoch + 1))
    accur_each_epoch  = 0
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train!
    loss = 0
    num_correct = 0
    tempt_num_correct = 0
    mark_time = time.time()
    count = 1
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
      if count == 50:
        print(
        '\r[Step %d] Past 50 steps: \033[33mTotal Average Loss %.3f | Accuracy: %.2f%%\033[0m' %
        (i + 1, loss / 100, num_correct/50*100)
        )
        loss = 0
        num_correct = 0
        count = 0
      elif i % 10 == 9:
        print(
          f"\r[Step %d] Past 10 steps: Average Loss %.3f | Correct Inference: %d" %
          (i + 1, loss / 100, num_correct), end=""
        )
      count+=1
      l, acc = train(im, label, Sequential, 16)
      loss += l
      num_correct += acc
      accur_each_epoch += acc
    print("\n\033[34mSummary !\033[0m: \033[32mAccuracy over previous epoch: ",accur_each_epoch/250 * 100)
    print("\033[0m")

def save_weight(Sequential):
   for layer in Sequential:
      layer.save_weight()

def load_weight(Sequential):
  for layer in Sequential:
     layer.load_weight_by_name()
def np_display(image):
  plt.imshow(image.astype(np.int32))  # 'gray' colormap for grayscale images
  plt.colorbar()  # Show color scale (optional)
  plt.show()

def test_accurancy_model(Sequential, testing_image, testing_label):
  correct_answer = 0
  total = 0
  for i in range(0,len(testing_image)):
    _,_,acc = forward(testing_image[i],testing_label[i],Sequential)
    total += 1
    correct_answer += acc
  return correct_answer/total

''' new_face_test_labels = np.zeros((64)).astype(np.int32)
for i in range(0,64):
  #new_face_test_labels[i] = np.amax(face_test_labels[i,0,:])
  #print(face_test_labels[i,0,:])
  new_face_test_labels = np.argmax(face_test_labels[i,0,:], axis=(0))
np.save('test_label.npy',new_face_test_labels) '''
#test_accurancy_model(Sequential=Sequential, testing_image=face_test_images, testing_label=face_test_labels)
''''
  This for traning.
'''
''' print('Starting Training !')
mark_time = time.time()
fit(Sequential=Sequential,train_images=new_face_train_image,train_labels=new_idex_face_train_label,epoch=10)
print("Traing time of FPGA take: ", time.time() - mark_time)
print(conv.backward_time)
print(second_conv.backward_time)
print(conv.forward_time)
print(second_conv.forward_time)
print("Accurancy on testing set: ",test_accurancy_model(Sequential=Sequential, testing_image=face_test_images, testing_label=face_test_labels))
save_weight(Sequential) '''

''''
  Used for testing a pre train model.
'''
''' load_weight(Sequential)
mark_time = time.time()
print("Accurancy on testing set: ",test_accurancy_model(Sequential=Sequential, testing_image=face_test_images, testing_label=face_test_labels))
print("Inference time: ", time.time() - mark_time)
print(conv.forward_time)
print(second_conv.forward_time)
print(pool.forward_time)
print(second_pool.forward_time) '''
'''
  Measure time of convolution
'''
def sliding_window_view_3d_reverse(arr, window_shape):
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
def new_conv_op(input,filters):
    windows_image = sliding_window_view_3d_reverse(input,(3,3))
    tempt = np.transpose(filters,(1,2,3,0))
    result = windows_image[:,:,:,:,:,np.newaxis]*tempt[np.newaxis,np.newaxis,:,:,:,:]
    result = np.sum(result, axis = (2,3,4))
    return result
def graph_create(function,num_sample,num_circle,name):
    kernel_size_axis = np.zeros(num_sample).astype(np.int32)
    time_exe_axis = np.zeros(num_sample).astype(np.float32)
    test_fitler = np.random.randint(-255, 255 ,size = (1,3,3,1))
    for i in range(5,5 + num_sample):
       test_image = np.random.randint(-1000, 1000, size = (i,i,1))
       tempt = timeit.Timer(functools.partial(function, test_image,test_fitler)) 
       total_time = (tempt.timeit(num_circle)/num_circle)
       kernel_size_axis[i-5] = i
       time_exe_axis[i-5] = total_time
    np.save(name + "_kernel_size_axis",kernel_size_axis)
    np.save(name + "_time_exe_axis", time_exe_axis)
    plt.plot(kernel_size_axis, time_exe_axis, marker='o', linestyle='-', color='b', label='Data')
    # Add labels and a title
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Graph of Two 1D Matrices')
    plt.legend()
    # Show the plot
    plt.grid(True)
    plt.show()
def graph_draw(kernel_size_axis_name, time_exe_axis_name):
    kernel_size_axis = np.load(kernel_size_axis_name)
    time_exe_axis = np.load(time_exe_axis_name)
    plt.plot(kernel_size_axis, time_exe_axis, marker='o', linestyle='-', color='b', label='Data')
    # Add labels and a title
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Graph of Two 1D Matrices')
    plt.legend()
    # Show the plot
    plt.grid(True)
    plt.show()
test_conv = Conv3x3(num_filters=1,num_chan=1, name="First_Conv",type_conv="new_conv"
               , fd=None,src_buffer=None,dest_buffer=None,kernel_buffer=None,num_signal=SIG_TEST, need_caculate_backprop=False, need_update_weight=True)     
'''
test_image =  np.random.randn(1000,1000,1).astype(np.float32) / 9
execution_time = timeit.Timer(functools.partial(test_conv.test_custom_conv2d, test_image,test_conv.filters))
total_time = execution_time.timeit(20)
print("Execution time of new_conv_op: ", total_time/20) '''

#graph_create(test_conv.new_conv_op, 200, 100, "Test")

graph_draw('./Test_kernel_size_axis.npy', './Test_time_exe_axis.npy')
#graph_draw('./New_test_kernel_size_axis.npy', './New_test_time_exe_axis.npy')
#print(f"Execution time for 1000 iterations: {execution_time:.6f} seconds")
#print(f"Average time per iteration: {execution_time / 100:.6f} seconds")             
conv.free_resource()
second_conv.free_resource()

