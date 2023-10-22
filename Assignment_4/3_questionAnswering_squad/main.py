import model

# loading dataset
dataset = model.data_loader(WholeDataset=[10,500])

# models
model_checkpoint = 'distilbert-base-uncased'
# model_checkpoint = 'xlm-roberta-base'

# defining processor
tokenizer = model.tokenizing(model_checkpoint)

# preparing train, validation, test
train_dataset, validation_dataset = model.preparing(dataset, tokenizer).prepare_dataset()

# defining model & trainer
net = model.Net(model_checkpoint)
trainer = model.training(net, train_dataset, validation_dataset, tokenizer).trainer()

# trainig the model
trainer.train()

# ploting the losses
plot = model.plots(trainer)
plot.loss()

# evaluation
model.evaluation(validation_dataset, trainer, tokenizer)

