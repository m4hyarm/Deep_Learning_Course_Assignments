import model

bijan_dir = '/kaggle/input/upc-2017/UPC-2017.txt/UPC-2016.txt'
glove_dir= '/kaggle/input/upc-2017/glove.6B.100d/glove.6B.100d.txt'

# reading and preprocessing data
preprocess = model.preprocessing(bijan_dir, glove_dir)
bijankhan_df = preprocess.bijan2dataframe()
words, embeds = preprocess.glove2array()

# maping words to index and vectors
dic = model.dictionary(bijankhan_df, words, embeds)
unique_adjs, adj2idx, idx2adj = dic.adjectives()
wrd2vec, wrd2idx, weights_matrix = dic.words()

# train & test & validation spliting
train, val, test = model.splitting(bijankhan_df, WholeDataset=True)

# loading data
traindata = model.MyDataset(bijankhan_df, wrd2idx, adj2idx)
valdata = model.MyDataset(bijankhan_df, wrd2idx, adj2idx)
testdata = model.MyDataset(bijankhan_df, wrd2idx, adj2idx)

# making batches of data
train_loader, val_loader, test_loader = model.myDataoader(traindata, valdata, testdata, 
                                                          wrd2idx, adj2idx, 
                                                          batch_size=64, numworkers=1)

# defining the network structure
net = model.RNN_POStagging(weights_matrix=weights_matrix, 
                           embedding_dim=weights_matrix.shape[1], hidden_dim=64, target_size=len(unique_adjs), 
                           model='lstm', padding_idx=wrd2idx['<pad>'], bidirect=False)

# defining loss function, optimizer, scheduler and their parameters
lossfunc, optimizer, lr_schedular = model.Net_parameters(net, ignore_index=wrd2idx['<pad>'], weight_decay=0.001, lr=0.0001)

# trainig the model
net, epochs_loss, epochs_acc, epochs_f1_micro, epochs_f1_macro = model.Training(train_loader, val_loader, 
                                                                                net, lossfunc, optimizer, lr_schedular, 
                                                                                wrd2idx, num_epochs = 20)

# print evaluations on test data
model.evaluation(test_loader, net, wrd2idx)

# ploting the losses
model.loss_plots(num_epochs=20,
                 train_loss=epochs_loss['train_loss'], 
                 val_loss=epochs_loss['val_loss'])

# ploting the scores
model.score_plots(num_epochs=20, 
                 epochs_acc=epochs_acc, epochs_f1_micro=epochs_f1_micro, epochs_f1_macro=epochs_f1_macro)
                 
# predictions
sentence = 'تشخیص ادات سخن ، یکی از مسائلی است که در پردازش متن به عنوان گام اولیه برای سایر کارها ، کاربرد بسیار دارد . '
model.predict_sentiment(sentence, net, wrd2idx, idx2adj)