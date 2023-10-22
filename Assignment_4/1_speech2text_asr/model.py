from datasets import load_dataset, load_metric, DatasetDict
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, HubertForCTC, TrainingArguments, Trainer
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# loading and preparing dataset
def data_loader(WholeDataset=True):
    if WholeDataset == True:
        tr, va, te = '', '', ''
    else:
        tr, va, te = WholeDataset
    train = load_dataset("superb", "asr", revision="2.2.2", split='train[:%s]'%tr)
    validation = load_dataset("superb", "asr", revision="2.2.2", split='validation[:%s]'%va)
    test = load_dataset("superb", "asr", revision="2.2.2", split='test[:%s]'%te)
    dataset = DatasetDict({'train': train, 'validation': validation, 'test': test})
    dataset = dataset.remove_columns(['speaker_id', 'chapter_id', 'id'])

    # lowercasing the letters
    def lowercase(batch):
        batch['text'] = batch['text'].lower()
        return batch
    dataset = dataset.map(lowercase)

    return dataset 



# making vaocabs dictionary
class vocabs_extraction():
    def __init__(self, dataset):
        self.dataset = dataset

    def extract_characters(self, batch):
        texts = " ".join(batch["text"])
        vocab = list(set(texts))
        return {'vocab': [vocab], 'text': [texts]}

    def vocabs_dic(self):
        vocabs = self.dataset.map(self.extract_characters, batched=True, 
                                  batch_size=-1, keep_in_memory=True, 
                                  remove_columns=['file', 'audio', 'text'])
        vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["validation"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["<unk>"] = len(vocab_dict)
        vocab_dict["<pad>"] = len(vocab_dict)
        with open('vocab.json', 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)



# defining a processor consist of a tokenizer and a feature extractor
def Wav2Vec_processor(vocabs='vocab.json', pretrained_feature_extractor='facebook/wav2vec2-base-960h',
                      unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|", 
                      padding_value=0, sampling_rate=16000):

    tokenizer = Wav2Vec2CTCTokenizer(vocab_file=vocabs, 
                                     unk_token=unk_token, 
                                     pad_token=pad_token, 
                                     word_delimiter_token=word_delimiter_token)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_feature_extractor, 
                                                                 padding_value=padding_value, 
                                                                 sampling_rate=sampling_rate)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, 
                                  tokenizer=tokenizer)

    return processor

    

# preparing dataset
class preparing():
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        
    def prepare(self, batch):
        audio = batch["audio"]

        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        
        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["text"]).input_ids
        return batch

    def prepare_dataset(self):
        prepared_dataset = self.dataset.map(self.prepare, remove_columns=self.dataset.column_names["train"])

        train_dataset = prepared_dataset['train']
        validation_dataset = prepared_dataset['validation']
        test_dataset = prepared_dataset['test']

        return train_dataset, validation_dataset, test_dataset



@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
 


# defining model
def Net(processor, pretrained_model='ntu-spml/distilhubert'):

    model = HubertForCTC.from_pretrained(pretrained_model, 
                                        ctc_loss_reduction="mean", 
                                        pad_token_id=processor.tokenizer.pad_token_id)
    model.freeze_feature_extractor()

    return model



# Dice score function
class training():
    def __init__(self, train_dataset, eval_dataset, processor, data_collator, model,
                 batchsize=16, num_epochs=5, lr=1e-3, weghtdecay=5e-3,
                 warmup_steps=250, logging_steps=100):

        self.model = model
        self.wer_metric = load_metric("wer")
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        self.feature_extractor = processor.feature_extractor
        self.data_collator = data_collator
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

    def compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)

        return {"WER": wer}

    def trainer(self):
        trainer = Trainer(model=self.model,
                        data_collator=self.data_collator,
                        args=self.training_args,
                        compute_metrics=self.compute_metrics,
                        train_dataset=self.train_dataset,
                        eval_dataset=self.eval_dataset,
                        tokenizer=self.feature_extractor,
                    )
        return trainer
    


# ploting the losses
def loss_plots(trainer):
    df = pd.DataFrame(trainer.state.log_history)
    eval = df.iloc[1::2]
    train = df.iloc[::2]
    ax = train.plot(x='step', y='loss', figsize=(11,6), label='Train', linewidth=2)
    eval.plot(x='step' , y='eval_loss', ax=ax, label='Validation', linewidth=2)
    plt.title('Train & Validation Loss', fontsize=20)
    plt.xlabel('Steps', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend(loc=9, fontsize=15)
    sns.despine()
    plt.show();


