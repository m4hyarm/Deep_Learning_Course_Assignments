import glob
import os
import model

dir = './trainSet/Stimuli/'

# getting calsses directories
classes_dir = glob.glob(os.path.join(dir,'**'))

# reading from each class and spliting train and test
train_dirs, test_dirs = model.dirs_loader(classes_dir, TrainToDataRatio=0.85, WholeDataset=True)

# loading images
traindata = model.SaliencyDataloader(train_dirs)
testdata = model.SaliencyDataloader(test_dirs)

# making batches of our images
train_loader, test_loader = model.myDataoader(traindata, testdata, batchsize=2, numworkers=1)

# defining the network structure
net = model.Net_Structure()

# initializing the weights
net.apply(lambda x: model.Init_Weights(x, std=0.00001))

# defining loss function, optimizer, scheduler and their parameters
lossfunc, optimizer, scheduler = model.Net_parameters(net, weight_decay=0.0001, lr=0.0001, momentum=0.9, 
                                                      step_size=10, gamma=0.1)

# trainig the model
net, epochs_loss, batches_loss = model.Training(train_loader, test_loader, 
                                                net, lossfunc, optimizer, scheduler, num_epochs = 1)

# ploting the losses
model.loss_plots(num_epochs=1,
                 train_loss=epochs_loss['train_loss'], 
                 test_loss=epochs_loss['test_loss'])

# ploting predictions
model.prediction_plots(net, test_loader)