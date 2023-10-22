import torchmetrics
from parsinorm import General_normalization
from tqdm import tqdm
import pandas as pd
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# preprocessing
class preprocessing():
    def __init__(self, bijan_dir, glove_dir):
        self.bijan_dir = bijan_dir
        self.glove_dir = glove_dir
        self.general_normalization = General_normalization()

    # word corrections
    def correction(self, word):
        general_normalization = self.general_normalization
        word = general_normalization.alphabet_correction(sentence=word)
        word = general_normalization.semi_space_correction(sentence=word)
        word = general_normalization.remove_comma_between_numbers(sentence=word)
        word = general_normalization.remove_repeated_punctuation(sentence=word)
        word = word.replace('أ','ا').replace('إ','ا').replace('اً','ا').replace('اٌ','ا').replace('اٍ','ا').replace('اَ','ا').replace('اُ','ا').replace('اِ','ا').replace('آ','ا')
        word = word.replace('ؤ','و').replace('ٍ','').replace('ٌ','').replace('ً','').replace('َ','').replace('ُ','').replace('ّ','').replace('ِ','')
        if word.isdigit():
            word = '0'
        return word

    def bijan2dataframe(self):
        sents = []
        adjs = []
        with open(self.bijan_dir, mode='r', encoding='utf-8') as file:
            text = file.readlines()
            word = []
            adj = []
            i = 0
            pbar = tqdm(total=len(text))
            while i < len(text):
                if text[i] != '\n':
                    line = text[i].strip().split('	')
                    a = line[-1].strip().upper()
                    w = self.correction(line[0])
                    if a != w:
                        adj.append(a)
                        word.append(w)
                else:
                    assert len(word) == len(adj)
                    sents.append(word)
                    adjs.append(adj)
                    word = []
                    adj = []

                i += 1
                pbar.update(1)

            print('\nLoop Completed')
            pbar.close()

        df = pd.DataFrame({'sentenceses':sents, 'adjectives':adjs})
        df.drop(index=437, inplace=True)
        return df
    
    def glove2array(self):
        words = []
        embeds = []
        with open(self.glove_dir, mode='r', encoding='utf-8') as file:
            text = file.readlines()
            for lines in tqdm(text):
                line = lines.split(' ')
                words.append(self.correction(line[0]))
                embeds.append(np.array(line[1:], dtype=np.float64))
        
        return words, embeds


# building dictionaries for index or vector mapping
class dictionary():
    def __init__(self, df, words, embeds):
        self.df = df
        self.words = words
        self.embeds = embeds

    def adjectives(self):
        unique_adjs = {'<pad>'}
        for sent in self.df.adjectives:
            for adj in sent:
                unique_adjs.add(adj)
        adj2idx = dict(zip(unique_adjs, range(len(unique_adjs))))
        idx2adj = {v: k for k, v in adj2idx.items()}
        return unique_adjs, adj2idx, idx2adj

    def words2(self):
        wrd2vec = dict(zip(self.words, self.embeds))
        wrd2vec['<pad>'] = np.zeros(300)
        for sent in self.df.sentenceses:
            for word in sent:
                if word not in wrd2vec.keys():
                    if word.endswith('ها') and word[:-2] in wrd2vec.keys():
                        wrd2vec[word] = wrd2vec[word[:-2]]
                    elif word.endswith('های') and word[:-3] in wrd2vec.keys():
                        wrd2vec[word] = wrd2vec[word[:-3]]
                    else:
                        wrd2vec[word] = wrd2vec['<unk>']

        wrd2idx = {}
        emb_dim = len(self.embeds[0])
        weights_matrix = torch.zeros((len(wrd2vec.keys()), emb_dim))
        for idx, word in enumerate(wrd2vec.keys()):
            weights_matrix[idx] = torch.from_numpy(wrd2vec[word])
            wrd2idx[word] = idx

        return wrd2vec, wrd2idx, weights_matrix


# train & test & validation spliting
def splitting(df, WholeDataset=True):
    if WholeDataset != True:
        df = df.sample(WholeDataset)
    train, val_test = train_test_split(df, test_size=0.3)
    val, test = train_test_split(val_test, test_size=0.5)
    train, val, test = train.reset_index(), val.reset_index(), test.reset_index()
    return train, val, test


# loading data
class MyDataset(data.Dataset):
    def __init__(self, df, wrd2idx, adj2idx):
        self.wrd2idx = wrd2idx
        self.adj2idx = adj2idx
        self.x_train=df.sentenceses.values
        self.y_train=df.adjectives.values
  
    def __getitem__(self,idx):
        x = self.x_train[idx]
        y = self.y_train[idx]

        x = [self.wrd2idx[word] for word in x]
        y = [self.adj2idx[adj] for adj in y]

        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return len(self.y_train)


# making batches of data
def myDataoader(traindata, valdata, testdata, wrd2idx, adj2idx, batch_size, num_workers=2):
    def padding(batch):
        texts, targets = zip(*batch)
        texts = pad_sequence(texts, batch_first=True, padding_value = wrd2idx['<pad>'] )
        targets = pad_sequence(targets, batch_first=True, padding_value = adj2idx['<pad>'])
        return texts, targets

    # defining dataloaders with desired batch size
    return (data.DataLoader(traindata, batch_size, shuffle=True, num_workers=num_workers, collate_fn=padding), 
            data.DataLoader(valdata, batch_size, shuffle=True, num_workers=num_workers, collate_fn=padding),
            data.DataLoader(testdata, batch_size, shuffle=True, num_workers=num_workers, collate_fn=padding))


# defining the residual network structure
class RNN_POStagging(nn.Module):
    def __init__(self, weights_matrix, embedding_dim, hidden_dim, target_size, model, padding_idx, bidirect=False):
        super(RNN_POStagging, self).__init__()

        self.word_embeddings = nn.Embedding.from_pretrained(weights_matrix, freeze=True, padding_idx=padding_idx) 
        
        if model == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirect)
        elif model == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirect)
        elif model == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirect)
        
        if bidirect:
            self.hidden2tag = nn.Linear(hidden_dim*2, target_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, target_size)
            
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        rnn_out, _ = self.rnn(embeds)
        tag_space = self.hidden2tag(rnn_out)
        return tag_space
        

# defining loss function, optimizer, scheduler and their parameters
def Net_parameters(net, ignore_index, weight_decay=0.0001, lr=0.001):
    
    # L2 loss
    lossfunc = nn.CrossEntropyLoss(ignore_index=ignore_index)
    # Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # decreasing the learning rate by factor of 0.1 if no improve seen on train loss for 2 steps
    lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=False)
    
    return lossfunc, optimizer, lr_schedular


# score function
class score():
    def __init__(self, wrd2idx, num_classes=32):
        self.torch_f1_micro = torchmetrics.F1Score(num_classes, average='micro')
        self.torch_f1_macro = torchmetrics.F1Score(num_classes, average='macro')
        self.wrd2idx = wrd2idx
        
    def f1(self, net, dataloader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Y, Y_preds = [], []
        for sent, tag in dataloader:
            preds = net(sent.to(device))
            Y_preds.append(torch.argmax(preds.reshape(-1,32), -1))
            Y.append(tag.reshape(-1))
        Y_preds, Y = torch.cat(Y_preds), torch.cat(Y)
        Y_preds, Y = Y_preds[Y != self.wrd2idx['<pad>']].to('cpu'), Y[Y != self.wrd2idx['<pad>']].to('cpu')
        return (self.torch_f1_micro(Y_preds, Y).item(), 
                self.torch_f1_macro(Y_preds, Y).item())
    

# trainig the model
def Training(traindata, valdata, model, lossfunc, optimizer, lr_schedular, wrd2idx, num_epochs = 20):

    startTime = time.time()
    num_epochs = num_epochs
    epochs_loss = {"train_loss": [], "val_loss": []}
    epochs_acc = {"train_acc": [], "val_acc": []}
    epochs_f1_micro = {"train_f1_micro": [], "val_f1_micro": []}
    epochs_f1_macro = {"train_f1_macro": [], "val_f1_macro": []}

    # moving the model to GPU if it's available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model.to(device)

    # calling dice and jaccard scores 
    torch_acc = torchmetrics.Accuracy()
    f1_score = score(wrd2idx, num_classes=32)


    for e in range(num_epochs):

        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0

        # train step
        print("EPOCH {}/{} :".format(e +1, num_epochs))
        with tqdm(traindata, desc ="  train") as t_data_train:

            net.train()
            train_counter = 0
            for sent, tag in t_data_train:
                sent, tag = sent.to(device), tag.to(device)
                
                # forward
                output = net(sent)
                loss = lossfunc(output.reshape(-1,32), tag.reshape(-1))
                o = torch.argmax(output.reshape(-1,32), -1)
                t = tag.reshape(-1)
                o, t = o[t != wrd2idx['<pad>']].to('cpu'), t[t != wrd2idx['<pad>']].to('cpu')
                acc = torch_acc(o, t)
                
                # set the parameter gradients to zero
                optimizer.zero_grad()

                # backward
                loss.backward()
                optimizer.step()
                
                # statistics
                train_loss += loss.item()
                train_acc += acc.item()

                train_counter += 1
                batchloss_train = train_loss/train_counter
                batchacc_train = train_acc/train_counter
                t_data_train.set_postfix(train_loss=batchloss_train, train_acc=batchacc_train, refresh=True)
        
        # validation step
        with torch.no_grad():
            with tqdm(valdata, desc ="    val") as t_data_eval:

                net.eval()
                val_counter=0
                for x, y in t_data_eval:
                    x, y = x.to(device), y.to(device)
                    
                    # forward
                    output = net(x)
                    loss = lossfunc(output.reshape(-1,32), y.reshape(-1))
                    acc = torch_acc(output.reshape(-1,32).to('cpu'), y.reshape(-1).to('cpu'))

                    # statistics
                    val_loss += loss.item()
                    val_acc += acc.item()

                    val_counter += 1
                    batchloss_val = val_loss/val_counter
                    batchacc_val = val_acc/val_counter
                    t_data_eval.set_postfix(val_loss=batchloss_val, val_acc=batchacc_val, refresh=True)

        # decreasing the LR with scheduler
        lr_schedular.step(batchloss_train)

        # scores
        train_f1_micro, train_f1_macro = f1_score.f1(net, traindata)
        val_f1_micro, val_f1_macro = f1_score.f1(net, valdata)
        print('F1_micro_train={:.3f},'.format(train_f1_micro), 'F1_micro_test={:.3f},'.format(val_f1_micro), 
              'F1_macro_train={:.3f},'.format(train_f1_macro), 'F1_macro_test={:.3f}'.format(val_f1_macro))

        epochs_loss["train_loss"].append(batchloss_train)
        epochs_loss["val_loss"].append(batchloss_val)
        epochs_acc["train_acc"].append(batchacc_train)
        epochs_acc["val_acc"].append(batchacc_val)
        epochs_f1_micro["train_f1_micro"].append(train_f1_micro)
        epochs_f1_micro["val_f1_micro"].append(val_f1_micro)
        epochs_f1_macro["train_f1_macro"].append(train_f1_macro)
        epochs_f1_macro["val_f1_macro"].append(val_f1_macro)


    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
    print("[INFO] Train Loss Reached: {}".format(epochs_loss["train_loss"][-1]))
    print("[INFO] Test Loss Reached: {}".format(epochs_loss["val_loss"][-1]))

    return net, epochs_loss, epochs_acc, epochs_f1_micro, epochs_f1_macro


# testing on test data
def evaluation(test_loader, net, wrd2idx):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loss = 0
    test_acc = 0
    i = 0
    lossfunc = nn.CrossEntropyLoss(ignore_index=wrd2idx['<pad>'])
    torch_acc = torchmetrics.Accuracy()
    f1_score = score(wrd2idx, num_classes=32)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = net.to(device)(x)
            testloss = lossfunc(output.reshape(-1,32), y.reshape(-1))
            testacc = torch_acc(output.reshape(-1,32).to('cpu'), y.reshape(-1).to('cpu'))
            test_loss += testloss.item()
            test_acc += testacc.item()
            i += 1
        
    # statistics
    test_loss = test_loss / i
    test_acc = test_acc / i
    test_f1_micro, test_f1_macro = f1_score.f1(net, test_loader)

    print('Model loss on test data: {}'.format(testloss))
    print('Model accuracy on test data: {:.2f} %'.format(test_acc*100))
    print('Model f1 micro on test data: {:.2f} %'.format(test_f1_micro*100))
    print('Model f1 macro on test data: {:.2f} %'.format(test_f1_macro*100))


# ploting the losses
def loss_plots(num_epochs, train_loss, val_loss):
    plt.figure(figsize=(15,8))
    plt.plot(range(1,num_epochs+1), train_loss, label='Train', linewidth=2)
    plt.plot(range(1,num_epochs+1), val_loss , label='Validation', linewidth=2)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(fontsize=25)
    plt.title('Train & Validation Loss', fontsize=30)
    plt.show();


# ploting the scores
def score_plots(num_epochs, epochs_acc, epochs_f1_micro, epochs_f1_macro):
    plt.figure(figsize=(25,5))
    plt.subplot(131)
    plt.plot(range(1,num_epochs+1), epochs_acc["train_acc"], label='Train')
    plt.plot(range(1,num_epochs+1), epochs_acc["val_acc"], label='Validation')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.legend(fontsize=20)
    plt.title('Train Accuracy', fontsize=30)
    plt.subplot(132)
    plt.plot(range(1,num_epochs+1), epochs_f1_micro["train_f1_micro"], label='Train')
    plt.plot(range(1,num_epochs+1), epochs_f1_micro["val_f1_micro"], label='Validation')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.legend(fontsize=20)
    plt.title('Train F1 Micro', fontsize=30)
    plt.subplot(133)
    plt.plot(range(1,num_epochs+1), epochs_f1_macro["train_f1_macro"], label='Train')
    plt.plot(range(1,num_epochs+1), epochs_f1_macro["val_f1_macro"], label='Validation')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.legend(fontsize=20)
    plt.title('Train F1 Macro', fontsize=30)
    plt.show();


# predictions
def predict_sentiment(sentence, net, wrd2idx, idx2adj):
    tokenized = [tok for tok in sentence.split()]
    indexed = []#[word2idx[t] for t in tokenized]
    for t in tokenized:
        if t in wrd2idx.keys():
            indexed.append(wrd2idx[t])
        else:
            indexed.append(wrd2idx['<unk>'])
    tensor = torch.LongTensor(indexed)
    prediction = net.to('cpu')(tensor)
    preds, ind= torch.max(F.softmax(prediction, dim=-1), 1)
    p = [idx2adj[idx] for idx in np.array(ind.to('cpu'))]
    for i, w in enumerate(tokenized):
        print('(', w, ',',p[i], ')')






