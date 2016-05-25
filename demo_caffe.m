% This demo shows how to find the adversarial perturbation for CaffeNet.
clear;

%Load MatCaffe
addpath('~/caffe/matlab'); % set path to caffe
caffe.set_mode_gpu(); %caffe.set_mode_cpu();
net_model = 'resources/deploy_caffenet.prototxt'; % CaffeNet's architecture
if ~exist('./bvlc_reference_caffenet.caffemodel','file')
    fprintf('Downloading model file...');
    websave('bvlc_reference_caffenet.caffemodel','http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel');       
end
net_weights = 'bvlc_reference_caffenet.caffemodel'; % CaffeNet's weights
phase = 'test'; % run with phase test (so that dropout isn't applied)

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

%Load mean image
d = load('resources/ilsvrc_2012_mean.mat');
mean_data = d.mean_data;
IMAGE_DIM = 256;

%Load Test images
image = imread('resources/test.jpg');

% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
image = image(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
image = permute(image, [2, 1, 3]);  % flip width and height
image = single(image);  % convert from uint8 to single
image = imresize(image, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
image = image - mean_data;  % subtract mean_data (already in W x H x C, BGR)
x = image(1:227,1:227,:);
    
[r,l_fooling,l_clean,itr] = adversarial_DeepFool_caffe(x,net);

caffe.reset_all(); % reset caffe

figure(1); 
x = permute(x+mean_data(1:227,1:227,:),[2,1,3]);
x = x(:,:,[3,2,1]);
r = permute(r,[2,1,3]);
r = r(:,:,[3,2,1]);
subplot(1,3,1); imagesc(x/256); title('Original image');
subplot(1,3,2); imagesc((x+r)/256); title('Perturbed image');
subplot(1,3,3); imagesc(r); title('Perturbation [scaled]');
