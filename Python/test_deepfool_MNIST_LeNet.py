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


class Net(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 6,
                               kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6,
                               out_channels = 16,
                               kernel_size = 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 2)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

# Load data
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.manual_seed(random_seed)  # Deterministic

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/', train=False, download=False,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.Resize(32),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
    batch_size=batch_size_test, shuffle=True)

# Reminding ourselves what this looks like
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data[1].shape)

# Network you're using (can change to whatever)
net = Net(10)
net.load_state_dict(torch.load("../models/MNIST/LeNet/model.pth"))

# Switch to evaluation mode
net.eval()

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

for batch_idx, (data, target) in enumerate(test_loader):
    r, loop_i, label_orig, label_pert, pert_image = deepfool(data, net)
    print(label_orig, label_pert)
    quit()
    for im, label in zip(data, target):
        r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)
        print(label_orig, label_pert)
        quit()













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
        im = torchvision.transforms.Compose([
                torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
                ])(orig_img)


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
