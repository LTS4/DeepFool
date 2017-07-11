# DeepFool
DeepFool is a simple algorithm to find the minimum adversarial perturbations in deep networks

### deepfool.py

This function implements the algorithm proposed in [[1]](http://arxiv.org/pdf/1511.04599) to find the adverasrial perturbation.

__Note__: The final softmax (loss) layer should be removed in order to prevent numerical instabilities.

The parameters of the wrapper are:

- `image`: Image of size `HxWx3d`
- `f`: feedforward function (input: images, output: values of activation **BEFORE** softmax).
- `grads`: gradient functions with respect to input (as many gradients as classes).
- `num_classes`: limits the number of classes to test against, by default = 10.

### test_deepfool.m

A simple demo which computes the adversarial perturbation for a test image from ImageNet dataset.

## Reference
[1] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard:
*DeepFool: a simple and accurate method to fool deep neural networks*.  In Computer Vision and Pattern Recognition (CVPR ’16), IEEE, 2016.
