import numpy as np
import time
def test_iterate_regions(image, num_filter):
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
def backprop(d_L_d_out, learn_rate, last_input, num_filters):
    '''
    Performs a backward pass of the conv layer.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    #d_L_d_filters = np.zeros(self.filters.shape)
    test_d_L_d_filters = np.zeros((8,3,3,8))
    #print(d_L_d_out.shape)
    tempt_sum = 0
    for im_region, check_im_region, i, j in test_iterate_regions(last_input,num_filters):
        #print(self.output_num_chan)
        #print(im_region.shape)
        #print(d_L_d_out.shape)
      # im_region has shape(3, 3, num_chan.) was cut from the last input image.
      #for f in range(self.num_filters):
        #print(d_L_d_filters[f,:,:,:].shape)
        #print(im_region.shape)
        #d_L_d_filters[f,:,:,:] has shape(3,3,num_chan)
        #d_L_d_out has shape()
        #
        test_time = time.time()
        #print(im_region.shape)
        #print([len(d_L_d_filters[f,:,0,0]),len(d_L_d_filters[f,0,:,0]),len(d_L_d_filters[f,0,0,:])])
        tempt = (d_L_d_out[i, j, :] * im_region)
        #print(tempt.shape)
        t = np.transpose(tempt, (3, 0, 1, 2))
        #print(t.shape)
        #print(self.output_num_chan)
        #print(d_L_d_filters[:,:,:,:].shape)
        #print(t.shape)
        test_d_L_d_filters[:,:,:,:] += t
        #d_L_d_out[i, j, f] * im_region
        ''' for f in range(self.num_filters):
        #print(d_L_d_filters[f,:,:,:].shape)
        #print(im_region.shape)
        #d_L_d_filters[f,:,:,:] has shape(3,3,num_chan)
        #d_L_d_out has shape()
          d_L_d_filters[f,:,:,:] += d_L_d_out[i, j, f] * check_im_region  '''
    ''' print("test_d_L_d_filters: ")
    print(test_d_L_d_filters[0,:,0,0])
    print("d_L_d_filters: ")
    print(d_L_d_filters[0,:,0,0])
    #print("Efficient time: ") '''
    return test_d_L_d_filters
    #print(time.time() - mark_time)
# Let's say we have a 4D image (height, width, depth1, depth2)
image = np.random.rand(30,30,8).astype(np.float32)  # Shape (h, w, d1, d2)
#prinhttps://www.onlinegdb.com/logint(image)
# kernel shape(x1,y1,2)
h, w, num_chan = image.shape
tempt_image = np.zeros(image.shape, dtype = np.float32)
tempt_image = image
expanded_tempt_image = np.expand_dims(tempt_image, axis=3)
expanded_tempt_image = np.tile(expanded_tempt_image, (1, 1, 1, 8))
d_L_d_out = np.random.rand(28,28,8).astype(np.float32)

# Filters has shape(3,3,2,2)
# Step 1: Create sliding windows over the height and width dimensions

# (x1,y1) are the coordinate of 3x3 submatrix in the s
# shape((x1,y1,3,3,2,2)) of windows
# (2,2) is the num of filter and subfilter that the 3x3 submatrix belong to
# (3,3) is the size of submatrix.
mark_time = time.time()
windows = np.lib.stride_tricks.sliding_window_view(expanded_tempt_image, (3, 3), axis=(0, 1))
windows = np.transpose(windows, (0,1,4,5,2,3))
mark_time = time.time()
test = windows * d_L_d_out[:, :, np.newaxis, np.newaxis, np.newaxis, :]
print(time.time() - mark_time)
test = np.sum(test, axis=(0,1))
print(time.time() - mark_time)
test = np.transpose(test, (3,0,1,2))

#test_tempt = d_L_d_out[3,3,:] * expanded_tempt_image[3:6,3:6,:,:]
mark_time = time.time()
test_backprop = backprop(d_L_d_out, 0.001, image, 8)
print(time.time() - mark_time)

print(np.amax(test - test_backprop))
#print(test[0,0,0,:])
#print(test_backprop[0,0,0,:])
#print(windows[0,0,:,:,0,1])
#print(kernel[0,0,1])
# Now 'windows' is a view with shape (height-2, width-2, 3, 3, d1, d2)
# where height-2 and width-2 are the valid regions for applying the 3x3 kernel

# Step 2: Apply the kernel using np.tensordot
# We perform the dot product along the 3x3 windows (axes 2 and 3) with the kernel
#output = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))

