import model

# loading dataset
dataset = model.data_loader(WholeDataset=True)

# defining feature extractor
feature_extractor = model.Wav2Vec_feature_extractor()

# preparing train, validation, test
train_dataset, validation_dataset, test_dataset = model.preparing(dataset, feature_extractor).prepare_dataset()

# defining model & trainer
net = model.Net()
trainer = model.training(net, train_dataset, validation_dataset, feature_extractor).trainer()

# trainig the model
trainer.train()

# ploting
plot = model.plots(trainer, model, test_dataset)
plot.loss()
plot.confusion_matrix()

# print evaluations on test data
print('Test Results:\n', trainer.evaluate(test_dataset, metric_key_prefix='test'), '\n')
