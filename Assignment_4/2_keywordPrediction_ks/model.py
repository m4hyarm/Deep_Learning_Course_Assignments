from datasets import load_dataset, load_metric, DatasetDict
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor, TrainingArguments, Trainer
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix


# loading and preparing dataset
def data_loader(WholeDataset=True):
    if WholeDataset == True:
        tr, va, te = '', '', ''
    else:
        tr, va, te = WholeDataset
    train = load_dataset("superb", "ks", revision="2.2.2", split='train[:%s]'%tr)
    validation = load_dataset("superb", "ks", revision="2.2.2", split='validation[:%s]'%va)
    test = load_dataset("superb", "ks", revision="2.2.2", split='test[:%s]'%te)
    dataset = DatasetDict({'train': train, 'validation': validation, 'test': test})

    # removing class 10 (silence) cause it has just 6 sample in train dataset
    dataset = dataset.filter(lambda x: x['label'] != 10)
    label_names = dataset["train"].features["label"].names
    label_names.remove('_silence_')
    for i in ['train', 'test', 'validation']:
        dataset[i].features['label'].num_classes = 11
        dataset[i].features['label'].names = label_names

    return dataset 



# defining a processor consist of a tokenizer and a feature extractor
def Wav2Vec_feature_extractor(pretrained_feature_extractor='ntu-spml/distilhubert'):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_feature_extractor)
    return feature_extractor

    

# preparing dataset
class preparing():
    def __init__(self, dataset, feature_extractor, max_duration=1):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration
        
    def prepare(self, batch):
        audio = batch["audio"]

        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = self.feature_extractor(audio["array"], 
                                                sampling_rate=audio["sampling_rate"], 
                                                max_length=int(audio["sampling_rate"]*self.max_duration),
                                                truncation=True).input_values[0]
        if batch['label'] == 11:
            batch['label'] = 10
    
        return batch

    def prepare_dataset(self):
        encoded_dataset = self.dataset.map(self.prepare, remove_columns=["audio", "file"])

        train_dataset = encoded_dataset['train']
        validation_dataset = encoded_dataset['validation']
        test_dataset = encoded_dataset['test']

        return train_dataset, validation_dataset, test_dataset



# defining model
def Net(pretrained_model='ntu-spml/distilhubert', num_labels=11):

    model = HubertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels)
    model.freeze_feature_extractor()

    return model



# training
class training():
    def __init__(self, model, train_dataset, eval_dataset, feature_extractor,
                 batchsize=64, num_epochs=10, lr=1e-3, weghtdecay=5e-3,
                 warmup_steps=250, logging_steps=300):

        self.model = model
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        self.feature_extractor = feature_extractor
        self.metrics_acc = load_metric("accuracy")
        self.metrics_pre = load_metric("precision")
        self.metrics_rec = load_metric("recall")
        self.training_args = TrainingArguments(output_dir='mahyar_hubert',
                                            per_device_train_batch_size=batchsize,
                                            per_device_eval_batch_size=batchsize,
                                            evaluation_strategy='steps',
                                            num_train_epochs=num_epochs,
                                            learning_rate=lr,
                                            weight_decay=weghtdecay,
                                            warmup_steps=warmup_steps,
                                            report_to='none',
                                            disable_tqdm=False,
                                            dataloader_num_workers=2,
                                            logging_steps=logging_steps,
                                            save_strategy='no',
                                            group_by_length=False,
                                            )

    def compute_metrics(self, eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return {'Accuracy':self.metrics_acc.compute(predictions=predictions, references=eval_pred.label_ids)['accuracy'],
                'Precision Micro':self.metrics_pre.compute(predictions=predictions, references=eval_pred.label_ids, average='micro')['precision'],
                'Precision Macro':self.metrics_pre.compute(predictions=predictions, references=eval_pred.label_ids, average='macro')['precision'],
                'Recall Micro':self.metrics_rec.compute(predictions=predictions, references=eval_pred.label_ids, average='micro')['recall'],
                'Recall Macro':self.metrics_rec.compute(predictions=predictions, references=eval_pred.label_ids, average='macro')['recall'],}


    def trainer(self):
        y = self.train_dataset['label']
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=np.array(y))
        class_weights = torch.tensor(class_weights,dtype=torch.float).to('cuda:0')
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')

        # replacing ctc loss by costum trainer
        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                # forward pass
                outputs = model(**inputs)
                logits = outputs.get("logits")
                # compute custom loss (suppose one has 3 labels with different weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer = CustomTrainer(model=self.model,
                        args=self.training_args,
                        compute_metrics=self.compute_metrics,
                        train_dataset=self.train_dataset,
                        eval_dataset=self.eval_dataset,
                        tokenizer=self.feature_extractor,
                    )
        return trainer
    



# ploting
class plots():
    def __init__(self, trainer, model, test_dataset):
        self.trainer = trainer
        self.model = model
        self.test_dataset = test_dataset

    def loss(self):
        df = pd.DataFrame(self.trainer.state.log_history)
        eval = df.iloc[1::2]
        train = df.iloc[::2]
        ax = train.plot(x='step', y='train_loss', figsize=(11,6), label='Train', linewidth=2)
        eval.plot(x='step' , y='eval_loss', ax=ax, label='Validation', linewidth=2)
        plt.title('Train & Validation Loss', fontsize=20)
        plt.xlabel('Steps', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.legend(loc=9, fontsize=15)
        sns.despine()
        plt.show();

    def confusion_matrix(self):
        def map_to_result(batch):
            with torch.no_grad():
                input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
                logits = self.model(input_values).logits
            batch["predictions"] = torch.argmax(logits, dim=-1)
            batch['label_ids'] = batch['label']
            return batch

        results = self.test_dataset.map(map_to_result, remove_columns=self.test_dataset.column_names)
        labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown']
        plt.figure(figsize=(12,10))
        sns.heatmap(confusion_matrix(results['label_ids'], results['predictions']), cmap='Blues', 
                    annot=True, cbar=False, fmt="d", linewidths=.5, yticklabels=labels, xticklabels=labels)
        plt.xlabel('Predicted Labels', fontsize=20, labelpad=25)
        plt.ylabel('True Labels', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15, rotation=0)
        plt.title('Confusion Matrix', fontsize=30)
        plt.show();