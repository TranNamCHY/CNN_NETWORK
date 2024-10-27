import ctypes
import numpy as np
import os
import signal
import time
import fcntl
import struct
import mmap
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
from tempt_softmax import tempt_Softmax
from flatten import Flatten
from dense import Dense

SET_PID_COMMAND = 0x40046401
PRE_SRC_BUFF = 0x40046402
PRE_KERNEL_BUFF = 0x40046403
PRE_DEST_BUFF = 0x40046404
SET_IMAGE_HEIGHT_WIDTH = 0x40046405
START_CACULATE = 0x40046406
MAX_DEST_BUFFER = 100*100
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

def forward(image, label, Sequential): 
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy array
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
  for layer in Sequential:
    out = layer.forward(out)
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0
  return out, loss, acc

def train(im, label, Sequential, lr=.005):
  '''
  Completes a full training step on the given image and label.
  Returns the cross-entropy loss and accuracy.
  - image is a 2d numpy array
  - label is a digit
  - lr is the learning rate
  '''
  # Forward
  out, loss, acc = forward(im, label,Sequential)

  # Calculate initial gradient
  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  # Backprop
  ''' gradient = t_softmax.backprop(gradient, lr)
  gradient = dense.backprop(gradient, lr)
  gradient = flatten.backprop(gradient)
  gradient = second_pool.backprop(gradient)
  gradient = second_conv.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  #print(gradient)
  gradient = conv.backprop(gradient, lr)
  #print("Output shape from conv2d: ") '''
  for layer in reversed(Sequential):
    gradient = layer.backprop(gradient, lr)
  return loss, acc

fd = os.open("/dev/device_axidma", os.O_RDWR)
set_pid(fd)
src_buffer,dest_buffer,kernel_buffer = prepare_mmap_buffer(fd)


# We only use the first 1k examples of each set in the interest of time.
# Feel free to change this if you want.
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#np.save('train_labels', train_labels)
#np.save('test_images', test_images)
#np.save('test_labels', test_labels)
train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')
train_images = train_images[0:1000].astype(np.float32)
#print(train_images[10])
train_images = np.expand_dims(train_images, axis = -1)
#print("After add a dimension: ")
#print(train_images[10,:,:,1])
test_image = np.zeros((28,28,1), dtype=np.float32)
train_labels = train_labels[0:1000]

test_images = test_images[0:1000]
test_labels = test_labels[0:1000]
test_image = train_images[0].astype(np.float32)

Sequential = []

conv1 = Conv3x3(num_filters=1,num_chan=1, name="First Conv",
                fd=fd,src_buffer=src_buffer, dest_buffer=dest_buffer, kernel_buffer=kernel_buffer, num_signal=(SIG_TEST))                 # 28x28x1 -> 26x26x32
conv1.filters[0,:,:,0] = [[1,2,3], [4,5,6], [7,8,9]]
#train_images[:,:,:] = 128
mark_time = time.time()
print(conv1.fpga_forward(train_images[0]/255 - 0.5)[:,:,0])
print("It take: ")
print(time.time() - mark_time)
print(conv1.int8_forward(train_images[0]/255 - 0.5)[:,:,0])
print("It take: ")
print(time.time() - mark_time)
'''Sequential.append(conv1)

pool = MaxPool2(name="First Maxpool")                  # 26x26x32 -> 13x13x32
Sequential.append(pool)

conv2 = Conv3x3(num_filters=16,num_chan=32, name="Second Conv",
                fd=fd,src_buffer=src_buffer, dest_buffer=dest_buffer, kernel_buffer=kernel_buffer, num_signal=(SIG_TEST + 1))  # 13x13x32 ->  11x11x16
Sequential.append(conv2)

second_pool = MaxPool2(name="Second Maxpool")  # 11x11x16 -> 5x5x16
Sequential.append(second_pool)

flatten = Flatten(name="Flatten") # 5x5x16 -> 5*5*16
Sequential.append(flatten)

dense1 = Dense(input_len=5 * 5 * 16, num_neuron=10, name="Dense1", need_update = True) # 5*5*16 -> 64
Sequential.append(dense1)

#dense2 = Dense(input_len=10, num_neuron=10, name="Dense2", need_update = True) # 64 -> 10
#Sequential.append(dense2)

t_softmax = tempt_Softmax(name="Softmax") # 10 -> 10
Sequential.append(t_softmax)

print('MNIST CNN initialized!')

# Train the CNN for 3 epochs
for epoch in range(3):
  print('--- Epoch %d ---' % (epoch + 1))

  # Shuffle the training data
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  # Train!
  loss = 0
  num_correct = 0
  mark_time = time.time()
  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i % 100 == 99:
      print("It take: ")
      print(time.time() - mark_time)
      mark_time = time.time()
      print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
      )
      loss = 0
      num_correct = 0

    l, acc = train(im, label, Sequential)
    loss += l
    num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)'''

conv1.free_resource()
#conv2.free_resource()
src_buffer.close()
dest_buffer.close()
kernel_buffer.close()
os.close(fd=fd)