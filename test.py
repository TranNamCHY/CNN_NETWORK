import numpy as np
import matplotlib.pyplot as plt
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
from tempt_softmax import tempt_Softmax
from flatten import Flatten
from dense import Dense
import ctypes
import numpy as np
import os
import signal
import time
import fcntl
import struct
import mmap
import warnings
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

fd = os.open("/dev/device_axidma", os.O_RDWR)
set_pid(fd)
src_buffer,dest_buffer,kernel_buffer = prepare_mmap_buffer(fd)

# We only use the first 1k examples of each set in the interest of time.
# Feel free to change this if you want.
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#np.save('train_labels', train_labels)
#np.save('test_images', test_images)
#np.save('test_labels', test_labels)
face_train_image = np.load('train_image.npy')
face_train_label = np.load('train_label.npy')
train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')
idex_face_train_label = np.load('idex_train_label.npy')
new_face_train_image = np.load('new_face_train_image.npy')
new_idex_face_train_label = np.load('new_idex_face_train_label.npy')
'''
new_face_train_image = np.zeros((250, 64, 64, 3)).astype(np.float32)
new_face_train_image[0:244,:,:] = face_train_image[:,:,:]
new_face_train_image[244:250] = face_train_image[0:6,:,:]
new_idex_face_train_label = np.zeros((250)).astype(np.int32)
new_idex_face_train_label[0:244] = idex_face_train_label[:]
new_idex_face_train_label[244:250] = idex_face_train_label[0:6]
np.save('new_face_train_image',new_face_train_image)
np.save('new_idex_face_train_label',new_idex_face_train_label) '''
train_images = train_images[0:1000].astype(np.float32)
test_images = test_images[0:1000].astype(np.float32)
#print(train_images[10])
train_images = np.expand_dims(train_images, axis = -1)
test_images = np.expand_dims(test_images, axis = -1)
# Display the matrix as an image
train_labels = train_labels[0:1000]
#print("After add a dimension: ")
#print(train_images[10,:,:,1])
test_image = np.zeros((28,28,1), dtype=np.float32)

#test_images = test_images[0:1000]
test_labels = test_labels[0:1000]
test_image = train_images[0].astype(np.float32)
#test_image[:,:,:] = 0
''' train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000] '''
Sequential = []

conv = Conv3x3(num_filters=8,num_chan=3, name="First_Conv",type_conv="int8bit_forward"
               , fd=fd,src_buffer=src_buffer,dest_buffer=dest_buffer,kernel_buffer=kernel_buffer,num_signal=SIG_TEST, need_caculate_backprop=False, need_update_weight=False)                  # 28x28x1 -> 26x26x16
Sequential.append(conv)                                                                                                                                      # 64x64x3 -> 62x62x16

pool = MaxPool2(name="First Maxpool")                  # 26x26x16 -> 13x13x16                                       
Sequential.append(pool)                                # 62x62x16 -> 31x31x16

second_conv = Conv3x3(num_filters=8,num_chan=8,name="Second_Conv",type_conv="int8bit_forward"
                      , fd=fd,src_buffer=src_buffer,dest_buffer=dest_buffer,kernel_buffer=kernel_buffer,num_signal=(SIG_TEST+1), need_caculate_backprop=True, need_update_weight=True) # 13x13x16 -> 11x11x16
Sequential.append(second_conv)                                                                                                                     # 31x31x16 -> 29x29x16                                                                       

second_pool = MaxPool2(name="Second Maxpool")  # 11x11x16 -> 5x5x16
Sequential.append(second_pool)                 # 29x29x16 -> 14x14x16

flatten = Flatten(name="Flatten") # 5x5x16 -> 13*13*32
Sequential.append(flatten)        

dense1 = Dense(input_len=14*14*8, num_neuron=16, name="Dense1", need_update = True) # 5*5*16 -> 64
Sequential.append(dense1)
#dense2 = Dense(input_len=64, num_neuron=16, name="Dense2", need_update = True) # 64 -> 10
#Sequential.append(dense2)
t_softmax = tempt_Softmax(name="Softmax") # 10 -> 10
Sequential.append(t_softmax)

''' Sequential1 = []

conv1 = Conv3x3(num_filters=16,num_chan=1, name="First Conv",type_conv="float32_forward", fd=fd,src_buffer=src_buffer,dest_buffer=dest_buffer,kernel_buffer=kernel_buffer,num_signal=SIG_TEST,need_caculate_backprop=False)                  # 28x28x1 -> 26x26x16
Sequential1.append(conv1)                                                                                                                                      # 64x64x3 -> 62x62x16

pool1 = MaxPool2(name="First Maxpool")                  # 26x26x16 -> 13x13x16                                       
Sequential1.append(pool1)                                # 62x62x16 -> 31x31x16

second_conv1 = Conv3x3(num_filters=16,num_chan=16,name="Second Conv",type_conv="float32_forward"
                      , fd=fd,src_buffer=src_buffer,dest_buffer=dest_buffer,kernel_buffer=kernel_buffer,num_signal=(SIG_TEST+1), need_caculate_backprop=True) # 13x13x16 -> 11x11x16
Sequential1.append(second_conv1)                                                                                                                     # 31x31x16 -> 29x29x16                                                                       

second_pool1 = MaxPool2(name="Second Maxpool")  # 11x11x16 -> 5x5x16
Sequential1.append(second_pool1)                 # 29x29x16 -> 14x14x16

flatten1 = Flatten(name="Flatten") # 5x5x16 -> 13*13*32
Sequential1.append(flatten1)        

dense12 = Dense(input_len=5*5*16, num_neuron=16, name="Dense1", need_update = True) # 5*5*16 -> 64
Sequential1.append(dense12)
#dense2 = Dense(input_len=10, num_neuron=10, name="Dense2", need_update = True) # 64 -> 10
#Sequential.append(dense2)
t_softmax1 = tempt_Softmax(name="Softmax") # 10 -> 10
Sequential1.append(t_softmax1) '''

def forward(image, label, Sequential): 
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy d
  - label is a digit
  '''
  # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
  # to work with. This is standard practice.
  ''' out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
  #print(out.dtype)
  out = second_conv.forward(out) 
  #print(out) 
  out = second_pool.forward(out)                       
  out = flatten.forward(out)
  out = dense.forward(out)
  out = t_softmax.forward(out)
  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0 '''

  out = image/255 - 0.5
  mark_time = time.time()
  for layer in Sequential:
    #mark_time = time.time()
    out = layer.forward(out)
    #print(layer.name + ":",time.time() - mark_time)
  #return out
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0
  return out, loss, acc

def predict(image, Sequential):
  out = image/255 - 0.5
  for layer in Sequential:
    #mark_time = time.time()
    out = layer.forward(out)
    #print(layer.name + ":",time.time() - mark_time)
  #return out
  return out
def train(im, label, Sequential, Output_neuron, lr=.005):
  '''
  Completes a full training step on the given image and label.
  Returns the cross-entropy loss and accuracy.
  - image is a 2d numpy array
  - label is a digit
  - lr is the learning rate
  '''
  # Forward
  #print("Forward time: ")
  #mark_time = time.time()de
  #print("Forward time : ")
  out, loss, acc = forward(im, label,Sequential)
  #print("Forward time: ")
  #print(time.time() - mark_time)
  # Calculate initial gradient
  gradient = np.zeros(Output_neuron)
  gradient[label] = -1 / out[label]

  # Backprop
  #total_mark_time = time.time()
  ''' gradient = t_softmax.backprop(gradient, lr)
  gradient = dense.backprop(gradient, lr)
  gradient = flatten.backprop(gradient)
  gradient = second_pool.backprop(gradient)
  gradient = second_conv.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  #print(gradient)
  gradient = conv.backprop(gradient, lr)
  #print("Output shape from conv2d: ") '''
  #print("Backward time : ")
  for layer in reversed(Sequential):
    #mark_time  = time.time()
    gradient = layer.backprop(gradient, lr)
    #print(layer.name + ":",time.time() - mark_time)
    #print(time.time() - mark_time)
  #print("Total time: ")
  #print(time.time() - total_mark_time)
  return loss, acc


def custom_maxpool_layer(input_tensor_image, pool_size):
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

def fit(Sequential,train_images, train_labels, epoch):
  accur_each_epoch  = 0
  for idex_epoch in range(epoch):
    print('--- Epoch %d ---' % (idex_epoch + 1))
    accur_each_epoch  = 0
    # Shuffle the training dataf
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
      '''if i % 100 == 99:
        #print("It take: ")
        #print(time.time() - mark_time)
        mark_time = time.time()
        print(
        '\r[Step %d] Past 100 steps: \033[33mTotal Average Loss %.3f | Accuracy: %d%%\033[0m' %
        (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0
      elif i % 10 == 9:
        print(
          f"\r[Step %d] Past 10 steps: Average Loss %.3f | Accuracy: %d%%" %
          (i + 1, loss / 100, num_correct), end=""
        ) '''
      if count == 50:
        #print("It take: ")
        #print(time.time() - mark_time)
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

def np_display(image):
  plt.imshow(image.astype(np.int32))  # 'gray' colormap for grayscale images
  plt.colorbar()  # Show color scale (optional)
  plt.show()

''' test_image = np.random.randn(28,28,1).astype(np.float32) 
test_image[:, :, :] = 5 '''
''' tempt_result = np.zeros((28*28*1), dtype = np.int32)
test_d_L_d_out = np.random.randn(26,26,16).astype(np.float32)
test_d_L_d_out[:,:,:] = 0.2
lib = ctypes.CDLL('./libmatrix.so')
conv.forward(test_image/255 - 0.5)
print(conv.filters)
print(conv.test_8bit_caculate_previos_error(test_d_L_d_out)) '''
'''
  Train test:
'''
''' mark_time = time.time()
train(im=train_images[0,:,:,:], label=0,Output_neuron=16, Sequential=Sequential1)
#print(forward(test_image/255 - 0.5, 0, Sequential))
print("Training with 26x26 image: ", time.time() - mark_time) '''
''' conv.type_conv = "float32_forward"
second_conv.type_conv = "float32_forward"
mark_time = time.time()
train(im=face_train_image[0,:,:,:], label=0,Output_neuron=16, Sequential=Sequential)
print("Training with 64x64 image: ", time.time() - mark_time) '''
''' mark_time = time.time()
train(im=face_train_image[0,:,:,:], Output_neuron=16, label=0, Sequential=Sequential)
print("Training with 64x64 image: ", time.time() - mark_time)
#print(forward(test_image/255 - 0.5, 0, Sequential)) '''

'''
  Forwarding test: 
'''
''' mark_time = time.time()
tempt1 = conv.forward(test_image/255 - 0.5)
print("fpga_forward take: ", time.time() - mark_time)
#print(tempt[:,:,0])

mark_time = time.time()
tempt2 = conv.float_forward(test_image/255 - 0.5)
print("int8bit_forward take: ", time.time() - mark_time)

test = np.amax(tempt1 - tempt2, axis = (0,1,2))
print("Max difference: ", test) '''
#print(tempt[:,:,0])
''' for i in range(0,16):
  conv.my_covolution_op(test_image[:,:,0]/255 - 0.5, conv.filters[0, :, :, 0])
print("my_covolution_op take: ",time.time() - mark_time) '''

''' mark_time = time.time()
conv.int8bit_forward(test_image/255 - 0.5)
print("int8bit_forward take: ",time.time() - mark_time) '''

#conv.tempt_forward(test_image)
#print("Dont care previous: ")
#mark_time = time.time()
#train(test_image,0, Sequential)
''' conv.filters[0,:,:,0] = 1
mark_time = time.time()
conv.my_covolution_op(test_image[:, :,0], conv.filters[0,:,:,0])
tempt = time.time() - mark_time
print("my_covolution_op take: ", tempt) '''


''' mark_time = time.time()
for i in range(0,16):
  tempt_mark_time = time.time()
  lib.convolution_2d_8bit(test_image[:, :,0].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), 28,28,
                        conv.filters[0,:,:,0].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                        tempt_result[:].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
  print(time.time() - tempt_mark_time)
print("convolution_2d_8bit take: ", time.time() - mark_time) '''
#print(conv.filters)
#print(test_d_L_d_out)
#print(tempt)
#print(tempt[:,:,0])
#print("Test: ")
#print(t[:,:,0])

''' mark_time = time.time()
tempt = second_conv.int8_forward(test_image)
print(time.time() - mark_time)
print("test 8bit: ")
print(tempt) '''


''' print(test_image.flatten())
print(test_image[:,:,0])
print(test_image[:,:,1])'''
#print(test_image[:,:,0])
''' mark_time = time.time()
tempt = pool.forward(test_image)
print(time.time() - mark_time)
mark_time = time.time()
tempt = pool.test_forward(test_image)
print(time.time() - mark_time) '''
#print(tempt[:,:,10])
print('MNIST CNN initialized!')
#face_train_image = np.load('new_train_image.npy').astype(np.float32)
#idex_face_train_label = np.load('new_idex_train_label.npy').astype(np.int32)
#mark_time = time.time()
#train(face_train_image[0,:,:,:], idex_face_train_label[0], Sequential, 16)
#print("It take: ", time.time() - mark_time)e1
''' image_path = "/home/nambcn/face_image/testing_image/fac4/4face14.jpg"
conv.filters = np.load('First_Conv_weight.npy')
second_conv.filters = np.load('Second_Conv_weight.npy')
dense1.biases = np.load ('Dense1_bias.npy')
dense1.weights = np.load('Dense1_weight.npy')
test_image = plt.imread(image_path)
np_display(test_image)
t = predict(test_image, Sequential)
print(np.round(t * 100,2)) '''
#conv.type_conv = "test_fpga_forward"
#mark_time = time.time()
#test_image = np.zeros((64,64,1)).astype(np.float32)
#test_image[:,:,0] = new_face_train_image[0,:,:,0]
#conv.forward(test_image)
#print("Forward time take: ", time.time() - mark_time)
#conv.int8bit_forward(face_train_image[0,:,:,:]/255 - 0.5)
#warnings.simplefilter('error', RuntimeWarning)
#fit(Sequential=Sequential,train_images=face_train_image,train_labels=idex_face_train_label,epoch=10)
#print("Training time take: ", time.time() - mark_time)
#mark_time = time.time()
#tempt1 = conv.forward(face_train_image[0,:,:,:])
#print("Training time take: ", time.time() - mark_time)
#conv.type_conv = "test_fpga_forward"
#mark_time = time.time()
#tempt2 = conv.forward(face_train_image[0,:,:,:])
#print("Training time take: ", time.time() - mark_time)
#print(np.amax(tempt1 - tempt2))
#print("Training time take: ", time.time() - mark_time)
#test_image = np.random.randn(66, 66, 3).astype(np.float32) / 9
#print(conv.filters[:,:,:,0])
#print(conv.filters[:,:,:,0])
#tempt2 = conv.test_fpga_backprop_custom_conv2d_8bit(test_image, conv.filters)
#tempt1 = conv.test_custom_conv2d_8bit(test_image, conv.filters)
#print(np.amax(tempt1 - tempt2))
#save_weight(Sequential)
#save_weight(Sequential)
'''
test_image = np.zeros((3,3,2)).astype(np.float32)
test_conv = Conv3x3(num_filters=2,num_chan=2, name="First_Conv",type_conv="float32_forward"
               , fd=fd,src_buffer=src_buffer,dest_buffer=dest_buffer,kernel_buffer=kernel_buffer,num_signal=SIG_TEST, need_caculate_backprop=True, need_update_weight=False) 
print(test_conv.forward(test_image))
test_d_L_d_out = np.zeros((1,1,2)).astype(np.float32)
test_d_L_d_out[:,:,0] = 0.1
test_d_L_d_out[:,:,1] = 0.2
print("Channel 1 of filters 0: ")
print(test_conv.filters[0,:,:,1])
print("Channel 1 of filters 1: ")
print(test_conv.filters[1,:,:,1])
tempt = test_conv.backprop(test_d_L_d_out, 0.001)
print("\\")
print(tempt[:,:,0])
print(tempt[:,:,1]) '''
mark_time = time.time()
fit(Sequential=Sequential,train_images=new_face_train_image,train_labels=new_idex_face_train_label,epoch=10)
print("Traing time take: ", time.time() - mark_time)
''' tempt = conv.filters
fit(Sequential=Sequential,train_images=face_train_image,train_labels=idex_face_train_label,epoch=6)
conv.type_conv = "int8bit_forward"
tempt1 = face_train_image[0,:,:,:]/255 - 0.5
tempt = conv1.filters
tempt1  = test_image/255 - 0.5
conv1.fpga_forward(tempt1)
print("Difference: ", np.amax(conv1.filters - tempt))
print("Difference: ", np.amax(conv1.last_input - tempt1))
#print("Difference: ", np.amax(conv.last_input - tempt1)) '''
''' plt.imshow(face_train_image[10].astype(np.int32))  # 'gray' colormap for grayscale images
plt.colorbar()  # Show color scale (optional)
plt.show()s
tempt = np.zeros((1,16), dtype = np.int32)
for i in range(0, 244):
   tempt += face_train_label[i, :, :] '''
''' mark_time = time.time() 
train(train_images[0],0,Sequential) '''



''' # Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
  _, l, acc = forward(im, label,Sequential)
  loss += l
  num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)
print('Test predict: ')
print(test_labels[10])
plt.imshow(test_images[10].astype(np.int32), cmap='gray')  # 'gray' colormap for grayscale images
plt.colorbar()  # Show color scale (optional)
plt.show() '''
conv.free_resource()
second_conv.free_resource()
src_buffer.close()
dest_buffer.close()
kernel_buffer.close()
os.close(fd) 

