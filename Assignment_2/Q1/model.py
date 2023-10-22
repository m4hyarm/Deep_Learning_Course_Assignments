import glob
import random
import os
import torch.utils.data as data
from torchvision import datasets, models, transforms
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# getting each class images directory and spliting train and test
def dirs_loader(dirs, TrainToDataRatio=0.85, WholeDataset=True):
    train_dirs = []
    test_dirs = []
    for dir in dirs:
        # getting all image addresses
        images = glob.glob(os.path.join(dir,'*.jpg'))

        # choosing a small amount of data or all of it
        if WholeDataset != True:
            images = random.sample(images, WholeDataset)

        # train & test spliting
        train_data = random.sample(images, k=int(TrainToDataRatio*len(images)))
        train_dirs += train_data
        test_dirs += (list(set(images) - set(train_data)))

    return train_dirs, test_dirs


# loading images
class SaliencyDataloader(data.Dataset):
    def __init__(self, dirs):
        self.img_files = dirs
        self.mask_files = []

        # finding mask directories with replacing the folder name
        for img_path in self.img_files:
            self.mask_files.append(img_path.replace('Stimuli', 'FIXATIONMAPS')) 

        # definig Transformations
        self.transfromers  = {'image':transforms.Compose([transforms.ToPILImage(),
                                                          transforms.Resize((240,320)), 
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))]),
                              'mask':transforms.Compose([transforms.ToPILImage(),
                                                         transforms.Resize((236,316)), 
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=0.5, std=0.5)])}
        
    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        # reading images with cv2 lib (output is BGR)
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0) # 0 is for greyscale
        image = self.transfromers['image'](image)
        mask = self.transfromers['mask'](mask)
        return image, mask

    def __len__(self):
        return len(self.img_files)


# making batches of our images
def myDataoader(traindata, testdata, batchsize=2, numworkers=1):

    # defining dataloaders with desired batch size
    return (data.DataLoader(traindata , batch_size=batchsize, shuffle=True, num_workers=numworkers), 
            data.DataLoader(testdata  , batch_size=batchsize, shuffle=True, num_workers=numworkers))


# defining the network structure
class Net_Structure(nn.Module):
    def __init__(self):
        super(Net_Structure, self).__init__()

        # defining network structure
        self.conv1 = nn.Conv2d(3,   48,  kernel_size=7,  stride=1, padding=3)
        self.norm = nn.LocalResponseNorm(5, 0.0001, 0.75)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(48,  128, kernel_size=5,  stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3,  stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=5,  stride=1, padding=2)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=5,  stride=1, padding=2)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=7,  stride=1, padding=3)
        self.conv7 = nn.Conv2d(128, 64,  kernel_size=11, stride=1, padding=5)
        self.conv8 = nn.Conv2d(64,  16,  kernel_size=11, stride=1, padding=5)
        self.conv9 = nn.Conv2d(16,  1,   kernel_size=13, stride=1, padding=6)
        self.deconv = nn.ConvTranspose2d(1,  1,   kernel_size=8,  stride=4, padding=2, bias=False)

    def forward(self, x):

        # defining forward steps
        x = F.relu(self.conv1(x))
        x = self.norm(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.deconv(x)
        return x
        

# initializing the weights
def Init_Weights(m, std=0.00001):

    # weight initializing the relu layers with He
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    # weight initializing the deconvolution layer with a normal distibution
    elif type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, mean=0, std=std)


# defining loss function, optimizer, scheduler and their parameters
def Net_parameters(net, weight_decay=0.0001, lr=0.0001, momentum=0.9, step_size=10, gamma=0.1):
    
    # L2 loss
    lossfunc = nn.MSELoss()
    # SGD optimizer
    optimizer = optim.SGD(net.parameters(), weight_decay=weight_decay, lr=lr, momentum=momentum)
    # decreasing the learning rate every 'step_size' steps by a factor of 'gamma'
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    return lossfunc, optimizer, scheduler



# trainig the model
def Training(traindata, valdata, model, lossfunc, optimizer, scheduler, num_epochs = 20):

    startTime = time.time()
    num_epochs = num_epochs
    epochs_loss = {"train_loss": [], "test_loss": []}
    batches_loss = {"train_loss": [], "test_loss": []}

    # moving the model to GPU if it's available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model.to(device)

    for e in range(num_epochs):

        trainloss = 0
        testloss = 0

        # train step
        print("EPOCH {}/{} :".format(e +1, num_epochs))
        with tqdm(traindata, desc ="  train") as t_data_train:

            net.train()
            train_counter = 0
            for image, mask in t_data_train:
                image, mask = image.to(device), mask.to(device)
                
                # forward
                output = net(image)
                trloss = lossfunc(output, mask)
                
                # set the parameter gradients to zero
                optimizer.zero_grad()

                # backward
                trloss.backward()
                optimizer.step()
                
                # statistics
                trainloss += trloss.item()*image.shape[0]
                train_counter += image.shape[0]
                batchloss_train = trainloss/train_counter
                t_data_train.set_postfix(LR=optimizer.param_groups[0]['lr'], train_loss=batchloss_train, refresh=True)

                batches_loss["train_loss"].append(batchloss_train)
        
        # validation step
        with torch.no_grad():
            with tqdm(valdata, desc ="    val") as t_data_eval:

                net.eval()
                test_counter=0
                for x, y in t_data_eval:
                    x, y = x.to(device), y.to(device)
                    
                    # forward
                    output = net(x)
                    teloss = lossfunc(output, y)

                    # statistics
                    testloss += teloss.item()*x.shape[0]
                    test_counter += x.shape[0]
                    batchloss_test = testloss/test_counter
                    t_data_eval.set_postfix(test_loss=batchloss_test, refresh=True)

                    batches_loss["test_loss"].append(batchloss_test)

        # decreasing the LR with scheduler
        scheduler.step()

        epochs_loss["train_loss"].append(batchloss_train)
        epochs_loss["test_loss"].append(batchloss_test)


    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
    print("[INFO] Train Loss Reached: {}".format(epochs_loss["train_loss"][-1]))
    print("[INFO] Test Loss Reached: {}".format(epochs_loss["test_loss"][-1]))

    return net, epochs_loss, batches_loss


# ploting the losses
def loss_plots(num_epochs, train_loss, test_loss):
    plt.figure(figsize=(13,5))
    plt.subplot(121)
    plt.plot(train_loss, 'r')
    plt.xticks(range(0,num_epochs), range(1,num_epochs+1))
    plt.xlabel('Epochs', fontsize=10, labelpad=8)
    plt.title('Train Loss', fontsize=25, pad=15)
    plt.subplot(122)
    plt.plot(test_loss , 'g')
    plt.xticks(range(0,num_epochs), range(1,num_epochs+1))
    plt.xlabel('Epochs', fontsize=10, labelpad=8)
    plt.title('Validation Loss', fontsize=25, pad=15)
    plt.tight_layout(pad=3)    
    plt.show;


# ploting predictions
def prediction_plots(net, testdata):
    # moving the network to cpu
    net.to('cpu')

    # definig a function that reverts transformations
    def invert(inp):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array(0.5)
        std = np.array(0.5)
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp

    fig, axes = plt.subplots(6, 3, figsize=(11,17), subplot_kw={'xticks': [], 'yticks': []})
    for i in [0,3,6,9,12,15]:
        img = next(iter(testdata))
        ax = axes.flat[i]
        ax.imshow(cv2.cvtColor(invert(img[0][0]), cv2.COLOR_BGR2RGB))
        ax = axes.flat[i+1]
        ax.imshow(invert(img[1][0]))
        ax = axes.flat[i+2]
        ax.imshow(invert(net(img[0][0]).detach()))

    axes.flat[0].set_title('Original', fontsize=20)
    axes.flat[1].set_title('Saliency', fontsize=20)
    axes.flat[2].set_title('Predicted', fontsize=20)

    plt.tight_layout(pad=0)
    plt.show();   




