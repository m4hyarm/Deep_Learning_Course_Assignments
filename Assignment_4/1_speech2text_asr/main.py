import model

# loading dataset
dataset = model.data_loader(WholeDataset=True)

# saving vocabs
model.vocabs_extraction(dataset).vocabs_dic()

# defining processor
processor = model.Wav2Vec_processor()

# preparing train, validation, test
train_dataset, validation_dataset, test_dataset = model.preparing(dataset, processor).prepare_dataset()

# defining model & trainer
net = model.Net(processor)
data_collator = model.DataCollatorCTCWithPadding(processor=processor, padding=True)
trainer = model.training(train_dataset, validation_dataset, processor, data_collator, net).trainer()

# trainig the model
trainer.train()

# ploting the losses
model.loss_plots(trainer)

# print evaluations on test data
print('Test Results:\n', trainer.evaluate(test_dataset, metric_key_prefix='test'), '\n')
