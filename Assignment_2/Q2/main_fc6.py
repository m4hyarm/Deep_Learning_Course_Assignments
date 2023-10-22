import glob
import os
import model
import random

dir = './tiny-imagenet-200/train/'

# getting calsses directories
classes = glob.glob(os.path.join(dir,'**'))

# selecting 20 calsses randomly
random.seed(476)
selected_classes_dir = random.sample(classes, 20)
selected_classes = [i.split('\\')[-1] for i in selected_classes_dir]
print('selected classes are:', selected_classes)

# reading from each class and spliting train and test
train_dirs, test_dirs = model.dirs_loader(selected_classes_dir, TrainToDataRatio=0.9, WholeDataset=True)

# loading images
traindata = model.InvertDataloader(train_dirs)
testdata = model.InvertDataloader(test_dirs)

# making batches of our images
train_loader, test_loader = model.myDataoader(traindata, testdata, batchsize=64, numworkers=1)

# defining the network structure
net = model.Net_fc6_Structure()

# defining loss function, optimizer, scheduler and their parameters
lossfunc, optimizer, scheduler = model.Net_parameters(net, weight_decay=0.0001, lr=0.001, 
                                                      betas=(0.9, 0.999), step_size=5, gamma=0.1)

# trainig the model
net, epochs_loss, batches_loss = model.Training(train_loader, test_loader, 
                                                net, lossfunc, optimizer, scheduler, num_epochs=1)

# ploting the losses
model.loss_plots(num_epochs=20, 
                 train_loss=epochs_loss['train_loss'], 
                 test_loss=epochs_loss['test_loss'])

# ploting predictions
model.prediction_plots(net, selected_classes_dir, train_dirs, test_dirs)