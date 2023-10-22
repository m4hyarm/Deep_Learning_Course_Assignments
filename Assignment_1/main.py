import pandas as pd
import numpy as np
import model

# reading the data
df = pd.read_excel('Dry_Bean_Dataset.xlsx')

# encoding the labels ( ordinal )
df['Class'] = df['Class'].map({'BARBUNYA':0, 'BOMBAY':1, 'CALI':2, 
                               'DERMASON':3, 'HOROZ':4, 'SEKER':5, 'SIRA':6})

# train and test spliting
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train, y_train, X_test, y_test = model.train_test_split(X, y, 0.7)

# definig the model structure
NN = model.NeuralNetwork(16, 50, 0, 7, std=1)

# fitting the model to data
fitted = NN.fit( X_train, y_train, X_test, y_test, normalization=True,
                 learning_rate=0.1, momentum=0.01, verbose=False, 
                 num_iters=600, batch_size=200 )

# reports
NN.reports()
NN.plots()
NN.confusion_matrix()
