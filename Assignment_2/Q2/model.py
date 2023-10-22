import glob
import random
import os
import torch.utils.data as data
from torchvision import datasets, models, transforms
from PIL import Image
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
def dirs_loader(dirs, TrainToDataRatio=0.9, WholeDataset=True):
    train_dirs = []
    val_dirs = []
    for dir in dirs:
        # getting all image addresses
        images = glob.glob(os.path.join(dir,'images','**'))

        # choosing a small amount of data or all of it
        if WholeDataset != True:
            images = random.sample(images, WholeDataset)

        # train & test spliting
        train_data = random.sample(images, k=int(TrainToDataRatio*len(images)))
        train_dirs += train_data
        val_dirs += (list(set(images) - set(train_data)))

    return train_dirs, val_dirs


# loading images
class InvertDataloader(data.Dataset):
    def __init__(self, dirs):
        self.img_files = dirs

        # definig Transformations
        self.transfromers  = transforms.Compose([transforms.Resize(227), 
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        img_path = self.img_files[index]

        # reading images with cv2 lib (output is BGR)
        image = Image.open(img_path).convert('RGB')
        image = self.transfromers(image)
        return image, image

    def __len__(self):
        return len(self.img_files)


# making batches of our images
def myDataoader(traindata, testdata, batchsize=64, numworkers=1):

    # defining dataloaders with desired batch size
    return (data.DataLoader(traindata , batch_size=batchsize, shuffle=True, num_workers=numworkers), 
            data.DataLoader(testdata  , batch_size=batchsize, shuffle=True, num_workers=numworkers))


# defining the network structure for reconstucting Conv2
class Net_conv2_Structure(nn.Module):
    def __init__(self):
        super(Net_conv2_Structure, self).__init__()

        # calling alexnet
        self.model = models.alexnet(pretrained=True)
        # freezing alexnet gradiants
        for param in self.model.parameters():
            param.requires_grad = False
        # definig reconstuction layers
        self.conv1 = nn.Conv2d(192, 256, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.upconv1 = nn.ConvTranspose2d(256, 256, 5, 2, 2, 1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 5, 2, 2, 1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 5, 2, 2, 1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, 5, 2, 2, 1)
        self.upconv5 = nn.ConvTranspose2d(32, 3, 5, 2, 2, 1)

    def forward(self, x):

        # defining forward steps
        x = self.model.features[:6](x)
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv3(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv4(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv5(x), 0.2, inplace=True)

        # Upsampling the output to input size
        if len(x.shape) == 4:
            x = F.interpolate(x, 227, mode='bilinear')
        elif len(x.shape) == 3:
            x = F.interpolate(x.view([1] + list(x.shape)), 227, mode='bilinear')
            x = x.view(3, 227, 227)

        return x


# defining the network structure for reconstucting Conv5
class Net_conv5_Structure(nn.Module):
    def __init__(self):
        super(Net_conv5_Structure, self).__init__()

        # calling alexnet
        self.model = models.alexnet(pretrained=True)
        # freezing alexnet gradiants
        for param in self.model.parameters():
            param.requires_grad = False
        # definig reconstuction layers
        self.conv1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.upconv1 = nn.ConvTranspose2d(256, 256, 5, 2, 2, 1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 5, 2, 2, 1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 5, 2, 2, 1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, 5, 2, 2, 1)
        self.upconv5 = nn.ConvTranspose2d(32, 3, 5, 2, 2, 1)

    def forward(self, x):

        # defining forward steps
        x = self.model.features(x)
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv3(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv4(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv5(x), 0.2, inplace=True)

        # Upsampling the output to input size
        if len(x.shape) == 4:
            x = F.interpolate(x, 227, mode='bilinear')
        elif len(x.shape) == 3:
            x = F.interpolate(x.view([1] + list(x.shape)), 227, mode='bilinear')
            x = x.view(3, 227, 227)

        return x


# defining the network structure for reconstucting Fc6
class Net_fc6_Structure(nn.Module):
    def __init__(self):
        super(Net_fc6_Structure, self).__init__()

        # calling alexnet
        self.model = models.alexnet(pretrained=True)
        self.model.classifier = self.model.classifier[:4]
        # freezing alexnet gradiants
        for param in self.model.parameters():
            param.requires_grad = False
        # definig reconstuction layers
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.upconv1 = nn.ConvTranspose2d(256, 256, 5, 2, 2, 1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 5, 2, 2, 1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 5, 2, 2, 1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, 5, 2, 2, 1)
        self.upconv5 = nn.ConvTranspose2d(32, 3, 5, 2, 2, 1)

    def forward(self, x):

        #  reshaping if input and output dimension diffrents
        if len(x.shape) == 3:
            x = x.view([1] + list(x.shape))

        # defining forward steps
        x = self.model(x)
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc3(x), 0.2, inplace=True)
        x = x.view(x.size(0), 256, 4, 4)
        x = F.leaky_relu(self.upconv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv3(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv4(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv5(x), 0.2, inplace=True)

        # Upsampling the output to input size & reshaping if input and output dimension diffrents
        x = F.interpolate(x, 227, mode='bilinear')
        if x.shape[0] == 1:
            x = x.view(3, 227, 227)

        return x



# defining the network structure for reconstucting Fc8
class Net_fc8_Structure(nn.Module):
    def __init__(self):
        super(Net_fc8_Structure, self).__init__()

        # calling alexnet
        self.model = models.alexnet(pretrained=True)
        # freezing alexnet gradiants
        for param in self.model.parameters():
            param.requires_grad = False
        # definig reconstuction layers
        self.fc1 = nn.Linear(1000, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.upconv1 = nn.ConvTranspose2d(256, 256, 5, 2, 2, 1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 5, 2, 2, 1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 5, 2, 2, 1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, 5, 2, 2, 1)
        self.upconv5 = nn.ConvTranspose2d(32, 3, 5, 2, 2, 1)

    def forward(self, x):

        #  reshaping if input and output dimension diffrents
        if len(x.shape) == 3:
            x = x.view([1] + list(x.shape))

        # defining forward steps
        x = self.model(x)
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc3(x), 0.2, inplace=True)
        x = x.view(x.size(0), 256, 4, 4)
        x = F.leaky_relu(self.upconv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv3(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv4(x), 0.2, inplace=True)
        x = F.leaky_relu(self.upconv5(x), 0.2, inplace=True)

        # Upsampling the output to input size and reshaping if input and output dimension diffrents
        x = F.interpolate(x, 227, mode='bilinear')
        if x.shape[0] == 1:
            x = x.view(3, 227, 227)

        return x



# defining loss function, optimizer, scheduler and their parameters
def Net_parameters(net, weight_decay=0.0001, lr=0.001, betas=(0.9, 0.999), step_size=5, gamma=0.1):
    
    # L2 loss
    lossfunc = nn.MSELoss()
    # ADAM optimizer for non-Alexnet layers
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), betas=betas, weight_decay=weight_decay, lr=lr)
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
def prediction_plots(net, selected_classes_dir, train_dirs, val_dirs):
    # moving the network to cpu
    net.to('cpu')

    # transformations
    transfromers  = transforms.Compose([transforms.Resize(227), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # definig a function that reverts transformations
    def invert(inp):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp

    # selecting 5 classes randomly 
    plt_smpls = [i.split('\\')[-1] for i in random.sample(selected_classes_dir, 5)]

    # sselecting a train and a test image from selected classes
    plt_train = []
    plt_val = []
    for i in plt_smpls:
        train_lst = [d for d in train_dirs if i in d]
        val_lst = [d for d in val_dirs if i in d]
        plt_train.append(train_lst[50])
        plt_val.append(val_lst[5])

    # ploting
    fig, axes = plt.subplots(10, 2, figsize=(5,23), subplot_kw={'xticks': [], 'yticks': []})
    n = 0
    for i in range(5):
        img = transfromers(Image.open(plt_train[i]).convert('RGB'))
        ax = axes.flat[n]
        ax.imshow(invert(img))
        ax = axes.flat[n+1]
        ax.imshow(invert(net(img).detach()))

        img = transfromers(Image.open(plt_val[i]).convert('RGB'))
        ax = axes.flat[n+2]
        ax.imshow(invert(img))
        ax = axes.flat[n+3]
        ax.imshow(invert(net(img).detach()))

        n += 4

    plt.tight_layout(pad=1)
    plt.show();   




