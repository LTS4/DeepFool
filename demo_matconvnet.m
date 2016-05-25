% This demo shows how to find the adversarial perturbation for an MNIST classifier

clear;
load('resources/sample_imdb');

opts.labels_limit = 10;%number of classes to be considered
opts.overshoot = 0.02; %overshoot for early termination

load('resources/net');
net = vl_simplenn_tidy(net); %add compatibility to newer versions of MatConvNet
net.layers(end) = [];

x = images.data;
[r,adversarial_label,clean_label,itr] = adversarial_DeepFool_matconvnet(x,net,opts);

figure(1); 
subplot(1,3,1); imagesc(x); colormap gray; title('Original image');
subplot(1,3,2); imagesc(x+r); colormap gray; title('Perturbed image');
subplot(1,3,3); imagesc(r); colormap gray; title('Perturbation [scaled]');
