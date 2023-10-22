import model

dir =  '/kaggle/input/lgg-mri-segmentation/kaggle_3m'

# train & test & validation spliting
train, val, test = model.dirs_loader(dir, TrainToDataRatio=0.8, WholeDataset=True)

# loading images
traindata = model.SegmentationDataloader(dir, train, augmentation=True)
valdata = model.SegmentationDataloader(dir, val)
testdata = model.SegmentationDataloader(dir, test)

# making batches of our images
train_loader, val_loader, test_loader = model.myDataoader(traindata, valdata, testdata, batchsize=16, numworkers=1)

# defining the network structure
net = model.ResidualUNet(in_channels=3, out_channels=1, init_features=32)

# defining loss function, optimizer, scheduler and their parameters
lossfunc, optimizer, lr_schedular = model.Net_parameters(net, weight_decay=0.001, lr=0.0001)

# trainig the model
net, epochs_loss, epochs_score = model.Training(train_loader, val_loader, 
                                                net, lossfunc, optimizer, lr_schedular, num_epochs = 40)

# print evaluations on test data
model.evaluation(test_loader, net)

# ploting the losses
model.loss_plots(num_epochs=40,
                 train_loss=epochs_loss['train_loss'], 
                 test_loss=epochs_loss['val_loss'])

# ploting predictions
model.prediction_plots(net, test_loader, 5).plot_img()