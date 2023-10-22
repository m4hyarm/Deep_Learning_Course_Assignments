from datasets import load_dataset, load_metric, DatasetDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator, TrainingArguments, Trainer
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from tqdm.auto import tqdm


# loading dataset
def data_loader(WholeDataset=True):
    if WholeDataset == True:
        tr, va  = '', ''
    else:
        tr, va  = WholeDataset
    train = load_dataset("squad_v2", split='train[:%s]'%tr)
    validation = load_dataset("squad_v2", split='validation[:%s]'%va)
    dataset = DatasetDict({'train': train, 'validation': validation})

    return dataset 



# defining a processor consist of a tokenizer and a feature extractor
def tokenizing(model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return tokenizer

    

# preparing dataset
class preparing():
    def __init__(self, dataset, tokenizer, max_length=512, doc_stride=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        
    def prepare_train_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_dataset(self):
        tokenized_datasets = self.dataset.map(self.prepare_train_features, remove_columns=self.dataset["train"].column_names, batched=True)

        train_dataset = tokenized_datasets['train']
        validation_dataset = tokenized_datasets['validation']

        return train_dataset, validation_dataset



# defining model
def Net(model_checkpoint):

    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    return model



# training
class training():
    def __init__(self, model, train_dataset, eval_dataset, tokenizer, data_collator=default_data_collator,
                 batchsize=8, num_epochs=3, lr=1e-5, weghtdecay=1e-4,
                 logging_steps=500):

        self.model = model
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.training_args = TrainingArguments(output_dir='mahyar',
                                            per_device_train_batch_size=batchsize,
                                            per_device_eval_batch_size=batchsize,
                                            evaluation_strategy='steps',
                                            num_train_epochs=num_epochs,
                                            learning_rate=lr,
                                            weight_decay=weghtdecay,
                                            report_to='none',
                                            disable_tqdm=False,
                                            dataloader_num_workers=2,
                                            logging_steps=logging_steps,
                                            save_strategy='no',
                                            group_by_length=False,
                                            )

    def trainer(self):
        trainer = Trainer(model=self.model,
                        args=self.training_args,
                        train_dataset=self.train_dataset,
                        eval_dataset=self.eval_dataset,
                        tokenizer=self.tokenizer,
                        data_collator=self.data_collator
                    )
        return trainer
    


# ploting the losses
class plots():
    def __init__(self, trainer):
        self.trainer = trainer

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



def evaluation(dataset, trainer, tokenizer):

    max_answer_length = 30
    n_best_size = 20
    squad_v2 = True
    max_length = 512
    doc_stride = 128

    for batch in trainer.get_eval_dataloader():
        break
    batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
    with torch.no_grad():
        output = trainer.model(**batch)
    output.start_logits.argmax(dim=-1), output.end_logits.argmax(dim=-1)

    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    validation_features = dataset.map(prepare_validation_features,
                                                    batched=True,
                                                    remove_columns=dataset.column_names)

    raw_predictions = trainer.predict(validation_features)

    validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))

    start_logits = output.start_logits[0].cpu().numpy()
    end_logits = output.end_logits[0].cpu().numpy()
    offset_mapping = validation_features[0]["offset_mapping"]
    # The first feature comes from the first example. For the more general case, we will need to be match the example_id to
    # an example index
    context = dataset[0]["context"]

    # Gather the indices the best start/end logits:
    start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
    end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
    valid_answers = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
            # to part of the input_ids that are not in the context.
            if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or offset_mapping[start_index] is None
                or offset_mapping[end_index] is None
            ):
                continue
            # Don't consider answers with a length that is either < 0 or > max_answer_length.
            if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                continue
            if start_index <= end_index: # We need to refine that test to check the answer is inside the context
                start_char = offset_mapping[start_index][0]
                end_char = offset_mapping[end_index][1]
                valid_answers.append(
                    {
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": context[start_char: end_char]
                    }
                )

    valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:n_best_size]

    examples = dataset
    features = validation_features

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)


    def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
        all_start_logits, all_end_logits = raw_predictions
        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        predictions = collections.OrderedDict()

        # Logging.
        print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_score = None # Only used if squad_v2 is True.
            valid_answers = []
            
            context = example["context"]
            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]

                # Update minimum null prediction.
                cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                            or offset_mapping[start_index] == []
                            or offset_mapping[end_index] == []
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue

                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append(
                            {
                                "score": start_logits[start_index] + end_logits[end_index],
                                "text": context[start_char: end_char]
                            }
                        )
            
            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}
            
            # Let's pick our final answer: the best one or the null answer (only for squad_v2)
            if not squad_v2:
                predictions[example["id"]] = best_answer["text"]
            else:
                answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
                predictions[example["id"]] = answer

        return predictions

    final_predictions = postprocess_qa_predictions(dataset, validation_features, raw_predictions.predictions)
    metric = load_metric("squad_v2")
    formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset]
    test_metrics = metric.compute(predictions=formatted_predictions, references=references)
    print(test_metrics)
