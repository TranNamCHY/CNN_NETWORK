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

conv = Conv3x3(num_filters=8,num_chan=3, name="First_Conv",type_conv="int8bit_forward"
               , fd=None,src_buffer=None,dest_buffer=None,kernel_buffer=None,num_signal=SIG_TEST, need_caculate_backprop=False, need_update_weight=True)                  # 28x28x1 -> 26x26x16
Sequential.append(conv)                                                                                                                                      # 64x64x3 -> 62x62x16

pool = MaxPool2(name="First Maxpool")                  # 26x26x16 -> 13x13x16                                       
Sequential.append(pool)                                # 62x62x16 -> 31x31x16

second_conv = Conv3x3(num_filters=8,num_chan=8,name="Second_Conv",type_conv="int8bit_forward"
                      , fd=None,src_buffer=None,dest_buffer=None,kernel_buffer=None,num_signal=(SIG_TEST+1), need_caculate_backprop=True, need_update_weight=True) # 13x13x16 -> 11x11x16
Sequential.append(second_conv)                                                                                                                     # 31x31x16 -> 29x29x16                                                                       

second_pool = MaxPool2(name="Second Maxpool")  # 11x11x16 -> 5x5x16
Sequential.append(second_pool)                 # 29x29x16 -> 14x14x16

flatten = Flatten(name="Flatten") # 5x5x16 -> 13*13*32
Sequential.append(flatten)        

dense1 = Dense(input_len=14*14*8, num_neuron=16, name="Dense1", need_update = True) # 5*5*16 -> 64
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
print('Starting Training !')
mark_time = time.time()
fit(Sequential=Sequential,train_images=new_face_train_image,train_labels=new_idex_face_train_label,epoch=10)
print("Traing time take: ", time.time() - mark_time)
print("Accurancy on testing set: ",test_accurancy_model(Sequential=Sequential, testing_image=face_test_images, testing_label=face_test_labels))
save_weight(Sequential)

''''
  Used for testing a pre train model.
'''
'''load_weight(Sequential)

np_display(face_test_images[6])
out = face_test_images[6]
out = out/255 - 0.5
for layer in Sequential:
    out = layer.forward(out)
np_display(out[:,:,3])

np_display(face_test_images[7])
out = face_test_images[7]
out = out/255 - 0.5
for layer in Sequential:
    out = layer.forward(out)
np_display(out[:,:,3]) '''

conv.free_resource()
second_conv.free_resource()

