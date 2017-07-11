# DeepFool
DeepFool is a simple algorithm to find the minimum adversarial perturbations in deep networks

__Note__: A Python implementation can be found [here](http://github.com/LTS4/universal/blob/master/python/deepfool.py).

### adversarial_perturbation.m

This function implements the algorithm proposed in [[1]](http://arxiv.org/pdf/1511.04599) to find the adverasrial perturbation.

### adversarial_DeepFool_matconvnet.m

This function is a wrapper to compute the adversarial perturbation for [MatConvNet](http://www.vlfeat.org/matconvnet/)'s models.

__Note__: The final softmax (loss) layer should be removed in order to prevent numerical instabilities.

### adversarial_DeepFool_caffe.m

This function is a wrapper to compute the adversarial perturbation for [Caffe](http://caffe.berkeleyvision.org)'s models.

__Note__: The final softmax (loss) layer should be removed in order to prevent numerical instabilities.

The arguments for both wrappers are:

- `x`: input image in `W*H*C` format
- `net`: network model where the last layer (loss) is removed.
- `opts`: a structure containing the following options:
  - `labels_limit`: limits the number of classes to test against (default = 10).
  - `overshoot`: used as a termination criterion to prevent vanishing updates (default = 0.02).
  - `max_iter`: maximum number of iterations (default = 100).
  - `norm_p`: determines which l_p norm to use (see [[1]](http://arxiv.org/pdf/1511.04599)) (default = 2).
  
### demo_matconvnet.m

A simple demo which computes the adversarial perturbation for an image from MNIST dataset for a LeNet classifier. The image is loaded from a standard IMDB file.

### demo_caffe.m

A simple demo which computes the adversarial perturbation for an image on CaffeNet classifier.

## Reference
[1] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard:
*DeepFool: a simple and accurate method to fool deep neural networks*.  In Computer Vision and Pattern Recognition (CVPR ’16), IEEE, 2016.
