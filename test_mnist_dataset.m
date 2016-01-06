% load caffe model and trained weights into MATLAB 
% input MNIST test dataset 
% output accuracy

clear
clc
close all
addpath mnistHelper
%%

% find the path to caffe on your computer and and set it here to make life easy
caffe_root='/usr/local/caffe/';

% Add caffe/matlab to your Matlab search PATH to use matcaffe
if exist([caffe_root,'matlab/+caffe'], 'dir')
  addpath([caffe_root,'matlab']);
else
  error('path does not exist');
end

% load model and weights
model = 'model/lenet.prototxt';
weights = 'model/lenet_iter_10000.caffemodel';

% set cafe mode
%caffe.set_mode_cpu();
caffe.set_mode_gpu();

% create net
net = caffe.Net(model, weights, 'test'); % create net and load weights

% load mnist data
filename=['dataset/t10k-images.idx3-ubyte'];
images = loadMNISTImages(filename);

% load labels
filename=['dataset/t10k-labels.idx1-ubyte'];
labels = loadMNISTLabels(filename);

% reshape 
I=reshape(images,28,28,1,size(images,2));

% permute
I1=permute(I,[2 1 3 4]);

% forwar to network
res = net.forward({I1});
prob = res{1};
[~,dig]=max(prob);

error=find(abs(dig-1-labels'),1)/10e4;
accuracy=(1-error)*100

