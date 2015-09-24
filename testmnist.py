
import numpy as np
import matplotlib.pyplot as plt
import sys
import os



# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples/caffe-test-mnist-jpg/

sys.path.insert(0, caffe_root + 'python')
import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


if not os.path.isfile(caffe_root + 'examples/mnist/lenet_iter_10000.caffemodel'):
    print("Cannot find trained caffemodel...")

# load trained network
caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'examples/mnist/lenet.prototxt',
                caffe_root + 'examples/mnist/lenet_iter_10000.caffemodel',
                caffe.TEST)


# set net to batch size of 50
net.blobs['data'].reshape(2,1,28,28)


# load image digit 
Xdata0= caffe.io.load_image(caffe_root + 'examples/images/zero.jpg')
Xdata1= caffe.io.load_image(caffe_root + 'examples/images/one.jpg')
Xdata7= caffe.io.load_image(caffe_root + 'examples/images/seven.jpg')

# make sure dimension is correct: 28*28*3
print('data dim:', Xdata0.shape,Xdata1.shape)


# feed data to the network
net.blobs['data'].data[...] =Xdata7[:,:,0]

# perfrom forward operation
out = net.forward()

print("Predicted class is #{}.".format(out['prob'].argmax()))



