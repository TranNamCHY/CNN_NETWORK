from tensorflow.keras.datasets import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
from tempt_softmax import tempt_Softmax
from flatten import Flatten
from dense import Dense
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images[0:1000].astype(np.float32)
#print(train_images[10])
train_images = np.expand_dims(train_images, axis = -1)
#print("After add a dimension: ")
#print(train_images[10,:,:,1])
test_image = np.zeros((28,28,1), dtype=np.float32)
train_labels = train_labels[0:1000]

train_images = train_images[0]
conv = Conv3x3(num_filters=8,num_chan=1)
print(conv.int8_forward(train_images/255 - 0.5))
print("Use the checked conv2d: ")
#rint(train_images)
print(conv.forward(train_images/255 - 0.5))
#print(conv.forward(test_images[0]))
#print(conv.forward(train_images/255 - 0.5))
#sprint(conv.forward(train_images[0]))s