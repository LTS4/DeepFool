from keras.layers import Input, Dense, Conv3D, MaxPooling3D, UpSampling3D
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

# The encoding process
input_img = Input(shape=(1, 3, 224, 224))  

def load_dataset():
    data_path = '../data/perturbed/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        #batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader, train_loader

tl, vl = load_dataset()


train_data = [data for batch_idx, (data, target) in enumerate(tl)]
target_data = [target for batch_idx, (data, target) in enumerate(tl)]
train_data,train_target,test_data,test_target = train_test_split(train_data,
                                                             target_data, 
                                                             test_size=0.2, 
                                                             random_state=13)


train_data = np.asarray([t.numpy() for t in train_data])
train_target = np.asarray([t.numpy() for t in train_target])
test_data = np.asarray([t.numpy() for t in test_data])
test_target=np.asarray([t.numpy() for t in test_target])
print(train_data)
print(train_data.shape)
print(type(train_data))


#Normalization
#not sure what to be reshaping it into
#train_data = train_data.reshape(-1,3, 224,224)
#test_data = test_data.reshape(-1, 3, 224,224)
train_data.shape, test_data.shape

train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)

train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,
                                                             train_data, 
                                                             test_size=0.2, 
                                                             random_state=13)


noise_factor = 0.5
x_train_noisy = train_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_X.shape)
x_valid_noisy = valid_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=valid_X.shape)
x_test_noisy = test_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test_data.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_valid_noisy = np.clip(x_valid_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

############
# Encoding #
############

# Conv1 #
x = Conv3D(filters = 16, kernel_size = (3, 3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling3D(pool_size = (2, 2, 2), padding='same')(x)

# Conv2 #
x = Conv3D(filters = 8, kernel_size = (3, 3, 3), activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size = (2, 2, 2), padding='same')(x) 

# Conv 3 #
x = Conv3D(filters = 8, kernel_size = (3, 3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling3D(pool_size = (2, 2, 2), padding='same')(x)

# Note:
# padding is a hyper-arameter for either 'valid' or 'same'. 
# "valid" means "no padding". 
# "same" results in padding the input such that the output has the same length as the original input.

############
# Decoding #
############

# DeConv1
x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(encoded)
x = UpSampling3D((2, 2, 2))(x)

# DeConv2
x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
x = UpSampling3D((2, 2, 2))(x)

# Deconv3
x = Conv3D(16, (3, 3, 3), activation='relu')(x)
x = UpSampling3D((2, 2, 2))(x)
decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

# Declare the model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Train the model


#autoencoder.fit(x_train, x_train,
#                epochs=100,
#                batch_size=128,
#                shuffle=True,
#                validation_data=(x_test, x_test)
#               )


autoencoder_train = autoencoder.fit(x_train_noisy, train_X,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_valid_noisy, valid_X)
               )         

print(autoencoder_train)      
pred = autoencoder.predict(x_test_noisy)


plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(10,20,1):
    plt.subplot(2, 10, i+1)
    plt.imshow(test_data[i, ..., 0], cmap='gray')
    curr_lbl = test_labels[i]
    #plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
plt.show()    
plt.figure(figsize=(20, 4))
print("Test Images with Noise")
for i in range(10,20,1):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test_noisy[i, ..., 0], cmap='gray')
plt.show()    

plt.figure(figsize=(20, 4))
print("Reconstruction of Noisy Test Images")
for i in range(10,20,1):
    plt.subplot(2, 10, i+1)
    plt.imshow(pred[i, ..., 0], cmap='gray')  
plt.show()
