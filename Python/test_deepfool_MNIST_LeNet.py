import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os

# Load data

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
    batch_size=batch_size_test, shuffle=True)

print("ok")
quit()

# Number of images to perturb
N = 118
# List to hold L2 norms of r for all perturbed images so rho can be caluclated at the end
r_arr = []
# List to hold original labels
orig_labels = []
# List to hold perturbed labels
pert_labels = []
# List to hold L2 norms
L2_norms = []
# List of original images
orig_imgs = []
# Cumulative sum for rho
rho_sum = 0

iter = 0

# Network you're using (can change to whatever)
net = models.googlenet(pretrained=True)

# Switch to evaluation mode
net.eval()

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

# Get list of files in ImageNet directory (you gotta save this in DeepFool/Python to get it to work like this)
for (root, dirs, files) in os.walk("ILSVRC2012_img_val", topdown=True):
    sorted_files = sorted(files, key=lambda item: int(item[18:23]))

# Now for every image:
for i in range(N):
    iter = iter + 1
    print("Iteration: ", iter)
    # Something wrong with this image, this is a patch fix
    if (sorted_files[i] != "ILSVRC2012_val_00000034.JPEG") and (sorted_files[i] != "ILSVRC2012_val_00000107.JPEG") and (sorted_files[i] != "ILSVRC2012_val_00000118.JPEG"):
        # Open image in directory (traverse from top down)
        orig_img = Image.open("ILSVRC2012_img_val/" + sorted_files[i])


        mean = [ 0.485, 0.456, 0.406 ]
        std = [ 0.229, 0.224, 0.225 ]

        # Get vector form of image so L2 norm of image x can be calculated (See denominator of eqn 15 in DeepFool paper)
        img_arr = np.array(orig_img)
        img_vect = img_arr.ravel()
        L2_norms.append(np.linalg.norm(img_vect))


        # Remove the mean
        im = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean,
                 std = std)])(orig_img)

        r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

        # Add L2 norm of perturbation to array (See numerator of eqn 15 in DeepFool paper)
        r_norm = np.linalg.norm(r)
        r_arr.append(r_norm)

        # Add original labels and perturbed labels to array (just in case you need them later, not rlly using rn)
        orig_labels.append(label_orig)
        pert_labels.append(label_pert)

        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

        str_label_orig = labels[np.int(label_orig)].split(',')[0]
        str_label_pert = labels[np.int(label_pert)].split(',')[0]

        # Add L2 norm of perturbation to array (See numerator of eqn 15 in DeepFool paper)
        r_norm = np.linalg.norm(r)
        r_arr.append(r_norm)

        # Add original labels and perturbed labels to array (just in case you need them later, not rlly using rn)
        orig_labels.append(label_orig)
        pert_labels.append(label_pert)

        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

        str_label_orig = labels[np.int(label_orig)].split(',')[0]
        str_label_pert = labels[np.int(label_pert)].split(',')[0]

        print("Original label = ", str_label_orig)
        print("Perturbed label = ", str_label_pert)
    
        clip = lambda x: clip_tensor(x, 0, 255)

        ### These commented lines were throwing errors

        #tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=map(lambda x: 1 / x, std)),
        #transforms.Normalize(mean=map(lambda x: -x, mean), std=[1, 1, 1]),
        #transforms.Lambda(clip),
        #transforms.ToPILImage(),
        #transforms.CenterCrop(224)])
    
        ### Changed it to this

        tf =  transforms.Compose([transforms.Normalize(mean = [0, 0, 0],
                           std = [(1/0.229), (1/0.244), (1/0.255)]), transforms.Normalize(mean = [-0.485, -0.456, -0.406], std=[1,1,1]),
              transforms.Lambda(clip), transforms.ToPILImage(),
              transforms.Resize(256),
              transforms.CenterCrop(224)])

        # Write image file to directory to hold perturbed images
        if (os.path.exists('pert_imgs') != 1):
            os.mkdir('pert_imgs')
            tf(pert_image.cpu()[0]).save('pert_imgs/' + sorted_files[i], 'JPEG')
    

        ## Commented this out because u probably don't want a bunch of images popping up
    
        #plt.figure()
        #plt.imshow(tf(pert_image.cpu()[0]))
        #plt.title(str_label_pert)
        #plt.show()

        # Add to cumulative sum term to get rho (See eqn 15 in DeepFool paper)
        rho_sum = rho_sum + r_norm / np.linalg.norm(img_vect)

# Compute average robustness (rho) for the simulation (See eqn 15 in DeepFool paper)
rho = (1/N)*rho_sum
print(rho)
