% This demo shows how to find the adversarial perturbation for an MNIST classifier

clear;
load('resources/sample_imdb');

opts.labels_limit = 10;%number of classes to be considered
opts.overshoot = 0.02; %overshoot for early termination

load('resources/net');
net.layers(end) = [];

x = images.data;
[r,adversarial_label,clean_label,itr] = adversarial_DeepFool_matconvnet(x,net,opts);

