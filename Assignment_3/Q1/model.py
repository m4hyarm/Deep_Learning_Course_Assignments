import glob
import random
import os
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict


# train & test & validation spliting
def dirs_loader(dir, TrainToDataRatio=0.8, WholeDataset=True):
    patients = next(os.walk(dir))[1]

    # choosing a small amount of data or all of it
    if WholeDataset != True:
        patients = random.sample(patients, WholeDataset)

    # train & test spliting
    random.seed(810100476)
    train = random.sample(patients, k=int(TrainToDataRatio*len(patients)))
    test_val = set(patients) - set(train)
    random.seed(810100476)
    val = random.sample(test_val, k=int(0.5*len(test_val)))
    test = set(test_val) - set(val)

    return train, val, test 


# loading images
class SegmentationDataloader(data.Dataset):
    def __init__(self, dir, patients, augmentation=False):
        self.mask_files = []
        self.aug = augmentation

        # transformations + augmentations
        self.transformers = [transforms.ToPILImage(),
                             transforms.Resize(256),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=0.5, std=0.5),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation((-15,15)),
                             transforms.RandomAffine(degrees=0, scale=(0.95,1.05)),
                             transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=15)], p=0.5)]
        
        # finding mask directories
        for p in patients:
            self.mask_files += glob.glob((os.path.join(dir, p, '*mask.tif')))

        # finding image directories with replacing the name 
        self.img_files = [img.replace('_mask', '') for img in self.mask_files] 
        
        
    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        # reading images with cv2 lib (output is BGR)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0) # 0 is for greyscale
        
        # normalizations (just input image)
        image = transforms.Compose(self.transformers[0:4])(image)
        mask = transforms.Compose(self.transformers[0:3])(mask)

        # augmentations
        if self.aug:
            t = transforms.Compose(self.transformers[4:])
            seed = np.random.randint(810100476)
            torch.manual_seed(seed)
            image = t(image)
            torch.manual_seed(seed)
            mask = t(mask)
         
        return image, mask

    def __len__(self):
        return len(self.img_files)


# making batches of our images
def myDataoader(traindata, valdata, testdata, batchsize=16, numworkers=1):

    # defining dataloaders with desired batch size
    return (data.DataLoader(traindata , batch_size=batchsize, shuffle=True, num_workers=numworkers), 
            data.DataLoader(valdata  , batch_size=batchsize, shuffle=True, num_workers=numworkers),
            data.DataLoader(testdata  , batch_size=batchsize, shuffle=True, num_workers=numworkers))


# defining the residual network structure
class ResidualUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(ResidualUNet, self).__init__()
        features = init_features

        # net structure
        # encoder
        self.encoder1 = ResidualUNet._block(in_channels, features, name="enc1")
        self.conv1x1_1 = nn.Conv2d(in_channels, features, kernel_size=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ResidualUNet._block(features, features * 2, name="enc2")
        self.conv1x1_2 = nn.Conv2d(features, features * 2, kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ResidualUNet._block(features * 2, features * 4, name="enc3")
        self.conv1x1_3 = nn.Conv2d(features * 2, features * 4, kernel_size=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ResidualUNet._block(features * 4, features * 8, name="enc4")
        self.conv1x1_4 = nn.Conv2d(features * 4, features * 8, kernel_size=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # bottleneck
        self.bottleneck = ResidualUNet._block(features * 8, features * 16, name="bottleneck")
        self.conv1x1_5 = nn.Conv2d(features * 8, features * 16, kernel_size=1)
        # decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = ResidualUNet._block((features * 8) * 2, features * 8, name="dec4")
        self.conv1x1_6 = nn.Conv2d((features * 8) * 2, features * 8, kernel_size=1)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = ResidualUNet._block((features * 4) * 2, features * 4, name="dec3")
        self.conv1x1_7 = nn.Conv2d((features * 4) * 2, features * 4, kernel_size=1)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = ResidualUNet._block((features * 2) * 2, features * 2, name="dec2")
        self.conv1x1_8 = nn.Conv2d((features * 2) * 2, features * 2, kernel_size=1)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = ResidualUNet._block(features * 2, features, name="dec1")
        self.conv1x1_9 = nn.Conv2d(features * 2, features, kernel_size=1)
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # forward steps
        # encoder
        enc1 = self.encoder1(x) + self.conv1x1_1(x)
        enc1 = F.relu(enc1)
        enc2 = self.encoder2(self.pool1(enc1)) + self.conv1x1_2(self.pool1(enc1))
        enc2 = F.relu(enc2)
        enc3 = self.encoder3(self.pool2(enc2)) + self.conv1x1_3(self.pool2(enc2))
        enc3 = F.relu(enc3)
        enc4 = self.encoder4(self.pool3(enc3)) + self.conv1x1_4(self.pool3(enc3))
        enc4 = F.relu(enc4)
        # bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4)) + self.conv1x1_5(self.pool4(enc4))
        bottleneck = F.relu(bottleneck)
        # decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4) + self.conv1x1_6(dec4)
        dec4 = F.relu(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3) + self.conv1x1_7(dec3)
        dec3 = F.relu(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2) + self.conv1x1_8(dec2)
        dec2 = F.relu(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1) + self.conv1x1_9(dec1)
        dec1 = F.relu(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv2d( in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "conv2", nn.Conv2d( in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )
        

# defining loss function, optimizer, scheduler and their parameters
def Net_parameters(net, weight_decay=0.001, lr=0.0001):
    
    # L2 loss
    lossfunc = nn.BCELoss()
    # Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # decreasing the learning rate by factor of 0.1 if no improve seen on train loss for 2 steps
    lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=False)
    
    return lossfunc, optimizer, lr_schedular


# Dice score function
class DiceScore(nn.Module):

    def __init__(self):
        super(DiceScore, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return dsc


# Jaccard score function
class JaccardScore(nn.Module):

    def __init__(self):
        super(JaccardScore, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() - intersection + self.smooth
        )
        return dsc
    

# trainig the model
def Training(traindata, valdata, model, lossfunc, optimizer, lr_schedular, num_epochs = 40):

    startTime = time.time()
    num_epochs = num_epochs
    epochs_loss = {"train_loss": [], "val_loss": []}
    epochs_score = { "train_dice": [], "val_dice": [], "train_jaccard": [], "val_jaccard": []}

    # moving the model to GPU if it's available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model.to(device)

    # calling dice and jaccard scores 
    dice = DiceScore()
    jaccard = JaccardScore()

    for e in range(num_epochs):

        train_loss = 0
        val_loss = 0
        train_dice = 0
        train_jacc = 0
        val_dice = 0
        val_jacc = 0

        # train step
        print("EPOCH {}/{} :".format(e +1, num_epochs))
        with tqdm(traindata, desc ="  train") as t_data_train:

            net.train()
            train_counter = 0
            for image, mask in t_data_train:
                image, mask = image.to(device), mask.to(device)
                
                # forward
                output = net(image)
                loss = lossfunc(output, mask)
                
                # set the parameter gradients to zero
                optimizer.zero_grad()

                # backward
                loss.backward()
                optimizer.step()
                
                # statistics
                train_dice += dice(torch.round(output), mask).item()
                train_jacc += jaccard(torch.round(output), mask).item()
                train_loss += loss.item()
                train_counter += 1
                t_data_train.set_postfix(train_loss=train_loss/train_counter, refresh=True)
        
        # validation step
        with torch.no_grad():
            with tqdm(valdata, desc ="    val") as t_data_eval:

                net.eval()
                val_counter=0
                for x, y in t_data_eval:
                    x, y = x.to(device), y.to(device)
                    
                    # forward
                    output = net(x)
                    loss = lossfunc(output, y)

                    # statistics
                    val_dice += dice(torch.round(output), y).item()
                    val_jacc += jaccard(torch.round(output), y).item()
                    val_loss += loss.item()
                    val_counter += 1
                    t_data_eval.set_postfix(val_loss=val_loss/val_counter, refresh=True)

        # decreasing the LR with scheduler
        lr_schedular.step(train_loss/train_counter)

        epochs_loss["train_loss"].append(train_loss/train_counter)
        epochs_loss["val_loss"].append(val_loss/val_counter)
        epochs_score["train_dice"].append(train_dice/train_counter)
        epochs_score["val_dice"].append(val_dice/val_counter)
        epochs_score["train_jaccard"].append(train_jacc/train_counter)
        epochs_score["val_jaccard"].append(val_jacc/val_counter)


    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
    print("[INFO] Train Loss Reached: {}".format(epochs_loss["train_loss"][-1]))
    print("[INFO] Test Loss Reached: {}".format(epochs_loss["val_loss"][-1]))

    return net, epochs_loss, epochs_score


# testing on test data
def evaluation(testloader, net):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Y = torch.tensor([])
    outputs = torch.tensor([])
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            output = net.to(device)(x)
            Y = torch.cat([Y,y.cpu().detach()])
            outputs = torch.cat([outputs,output.cpu().detach()])
        
    # statistics
    lossfunc = nn.BCELoss()
    dice = DiceScore()
    jaccard = JaccardScore()
    testloss = lossfunc(outputs, Y).item()
    test_dice = dice(outputs, Y).item()
    test_jacc = jaccard(outputs, Y).item()

    print('Model loss on test data: {}'.format(testloss))
    print('Model dice score on test data: {:.2f} %'.format(test_dice*100))
    print('Model jaccard score on test data: {:.2f} %'.format(test_jacc*100))


# ploting the losses
def loss_plots(num_epochs, train_loss, test_loss):
    plt.figure(figsize=(15,8))
    plt.plot(range(1,num_epochs+1), train_loss, linewidth=2)
    plt.plot(range(1,num_epochs+1), test_loss , linewidth=2)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(fontsize=25)
    plt.title('Train & Validation Loss', fontsize=30)
    plt.show();


# ploting predictions
class prediction_plots():

    def __init__(self, net, testdata, no_):
        # moving the network to cpu
        self.net = net.to('cpu')
        self.dataloaders = testdata
        self.no_ = no_

    # converting tensor to image (reverse normalization)
    def image_convert(self, image):
        image = image.clone().cpu().numpy()
        image = image.transpose((1,2,0))
        mean = np.array(0.5)
        std = np.array(0.5)
        image = std * image + mean
        return image

    def mask_convert(self, mask):
        mask = mask.clone().cpu().detach().numpy()
        return np.squeeze(mask)

    def plot_img(self):
        images,masks = next(iter(self.dataloaders))
        preds = self.net(images).to('cpu')
        preds = torch.round(preds)
        no_ = self.no_
        fig, axs = plt.subplots(3, no_, figsize=(12,7), sharey=True, subplot_kw={'xticks':[], 'yticks':[]})
        idx = 0
        for idx in range(no_):
            image = self.image_convert(images[idx])
            ax = axs.flat[idx]
            ax.imshow(image)
            ax.set_ylabel('MRI', fontsize=18)

            mask = self.mask_convert(masks[idx])
            ax = axs.flat[idx+no_]
            ax.imshow(mask,cmap='gray')
            ax.set_ylabel('Ground Truth', fontsize=18)
            
            pred = self.mask_convert(preds[idx])
            ax = axs.flat[idx+2*no_]
            ax.imshow(pred,cmap='gray')
            ax.set_ylabel('Predicted', fontsize=18)
                        
        for ax in axs.flat:
            ax.label_outer()
        plt.tight_layout(pad=0)
        plt.show()






