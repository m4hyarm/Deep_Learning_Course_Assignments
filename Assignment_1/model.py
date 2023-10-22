import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_test_split(X, y, ratio):

    # shuffeling the data order
    idx = [i for i in range(len(X))]
    np.random.seed(476)
    np.random.shuffle(idx)
    X = np.array(X)[idx]
    y = np.array(y)[idx]

    # splitting train and test
    msk = int(ratio*len(X))
    X_train = X[:msk]
    y_train = y[:msk]
    X_test = X[msk:]
    y_test = y[msk:]

    return X_train, y_train, X_test, y_test

class NeuralNetwork():

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, std=0.01):
        
        np.random.seed(476)

        # initializing the weights with given std and net structure
        self.parameters = {}
        self.parameters['W1'] = np.random.normal(0, std, (input_size, hidden_size1))
        self.parameters['b1'] = np.zeros(hidden_size1,)
        
        # choosing betwin one or two hidden layer
        self.hidden_size2 = hidden_size2
        if self.hidden_size2 == 0:
            self.parameters['W2'] = np.random.normal(0, std, (hidden_size1, output_size))
            self.parameters['b2'] = np.zeros(output_size,)
        else:
            self.parameters['W2'] = np.random.normal(0, std, (hidden_size1, hidden_size2))
            self.parameters['b2'] = np.zeros(hidden_size2,)
            self.parameters['W3'] = np.random.normal(0, std, (hidden_size2, output_size))
            self.parameters['b3'] = np.zeros(output_size,)

    def normalize(self, X):
        return (X - np.mean(X, axis = 0)) / np.std(X, axis = 0) 

    def relu(self, x, d = False):
        # d is for when we want derivatives
        if d:
            return 1*(x > 0)
        return np.maximum(0, x)
    

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True) # this part will help us to escape the overflow error
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

        
    def cross_entropy_cost(self, y_hat, y):
        return np.mean(-np.log(y_hat[range(len(y_hat)), y] + 1e-7 ))
        
        
    def softmax_crossentropy_derivatives(self, y_hat, y):
        # this function calculate derivtives of both softmax and cross entropy 
        N = y_hat.shape[0]
        dscores = y_hat.copy()
        dscores[range(N), y] -= 1
        dscores /= N
        return dscores
    
        
    def forward(self, X, y):
        
        self.o1 = X.dot(self.parameters['W1']) + self.parameters['b1']
        self.relu_out1 = self.relu(self.o1)
        self.o2 = self.relu_out1.dot(self.parameters['W2']) + self.parameters['b2']
        
        # again choosing betwin one or two hidden layer
        if self.hidden_size2 != 0:
            self.relu_out2 = self.relu(self.o2)
            self.o3 = self.relu_out2.dot(self.parameters['W3']) + self.parameters['b3']
            self.softmax_out = self.softmax(self.o3)
        else:
            self.softmax_out = self.softmax(self.o2)
        
        loss = self.cross_entropy_cost(self.softmax_out, y)
        return loss
        

    def backpropagation(self, X, y):
        
        grads = {}
        
        dscores = self.softmax_crossentropy_derivatives(self.softmax_out, y)

        # again choosing betwin one or two hidden layer
        if self.hidden_size2 == 0:
            grads['W2'] = self.relu_out1.T.dot(dscores)
            grads['b2'] = np.sum(dscores, axis = 0)
            grads['W1'] = np.dot((np.dot((dscores),self.parameters['W2'].T) * self.relu(self.o1, d = True)).T,X).T
            grads['b1'] = np.sum(np.dot((dscores),self.parameters['W2'].T) * self.relu(self.o1, d = True),axis = 0)
            
        else:
            grads['W3'] = self.relu_out2.T.dot(dscores)
            grads['b3'] = np.sum(dscores, axis = 0)
            grads['W2'] = np.dot((np.dot((dscores),self.parameters['W3'].T) * self.relu(self.o2, d = True)).T,self.relu_out1).T
            grads['b2'] = np.sum(np.dot((dscores),self.parameters['W3'].T) * self.relu(self.o2, d = True),axis = 0)
            grads['W1'] = np.dot((np.dot(np.dot((dscores),self.parameters['W3'].T)*self.relu(self.o2, d = True),self.parameters['W2'].T)*self.relu(self.o1, d = True)).T,X).T
            grads['b1'] = np.sum((np.dot(np.dot((dscores),self.parameters['W3'].T)*self.relu(self.o2, d = True),self.parameters['W2'].T)*self.relu(self.o1, d = True)),axis = 0)
        
        return grads
        

    def fit(self, X, y, X_test, y_test, normalization=True, learning_rate=1e-3, momentum=0, num_iters=100, batch_size=200, verbose=False):
        
        # this will nomalize the input data if we want
        if normalization:
            X, X_test = self.normalize(X), self.normalize(X_test)

        self.train_loss_history = []
        self.test_loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
        self.y_test_pred_history = []
        self.y_test = y_test

        for it in range(num_iters):
            
            idx = [i for i in range(X.shape[0])]
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]
            
            # these terms are for difining the momentum
            dW3 = 0
            db3 = 0
            dW2 = 0
            db2 = 0
            dW1 = 0
            db1 = 0

            for bch in range(int(np.ceil(X.shape[0] / batch_size))):

                start = bch * batch_size
                end = (bch+1) * batch_size

                X_batch, y_batch = X[start:end], y[start:end]

                loss  = self.forward(X_batch, y_batch)
                grads = self.backpropagation(X_batch, y_batch)

                self.parameters['W2'] += - learning_rate * grads['W2'] + momentum * dW2
                self.parameters['b2'] += - learning_rate * grads['b2'] + momentum * db2
                self.parameters['W1'] += - learning_rate * grads['W1'] + momentum * dW1
                self.parameters['b1'] += - learning_rate * grads['b1'] + momentum * db1
                
                # again choosing betwin one or two hidden layer
                if self.hidden_size2 != 0:
                    self.parameters['W3'] += - learning_rate * grads['W3'] + momentum * dW3
                    self.parameters['b3'] += - learning_rate * grads['b3'] + momentum * db3
                    dW3 = learning_rate * grads['W3'] +  momentum * dW3
                    db3 = learning_rate * grads['b3'] +  momentum * db3
                
                dW2 = learning_rate * grads['W2'] +  momentum * dW2
                db2 = learning_rate * grads['b2'] +  momentum * db2
                dW1 = learning_rate * grads['W1'] +  momentum * dW1
                db1 = learning_rate * grads['b1'] +  momentum * db1
                
            
            if verbose & (it%10 == 0):
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
                
                
            y_test_pred = self.predict(X_test)
            train_acc = np.mean(self.predict(X) == y)            
            test_acc = np.mean(y_test_pred == y_test)
            train_loss = self.forward(X, y)
            test_loss = self.forward(X_test, y_test)
            
            self.train_loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)
            self.train_acc_history.append(train_acc*100)
            self.test_acc_history.append(test_acc*100)
            self.y_test_pred_history.append(y_test_pred)
                    
        
        return { 
                'train_loss_history': self.train_loss_history,
                'test_loss_history': self.test_loss_history,
                'test_loss': self.test_loss_history[-1],
                'train_acc_history': self.train_acc_history,
                'test_acc_history': self.test_acc_history,
                'test_acc': self.test_acc_history[-1],
                'y_test_pred_history': self.y_test_pred_history,
                }

    
    def reports(self):
        print('test accuracy =', self.test_acc_history[-1], '%\n', 'test loss =',self.test_loss_history[-1])
        
        
    def plots(self):
        fig, ax = plt.subplots(2, 2, figsize=(15,10))
        ax1 = ax[0,0]
        ax1.set_title('train loss')
        ax1.plot(self.train_loss_history)
        ax2 = ax[0,1]
        ax2.set_title('test loss')
        ax2.plot(self.test_loss_history)
        ax3 = ax[1,0]
        ax3.set_title('train acc')
        ax3.plot(self.train_acc_history)
        ax4 = ax[1,1]
        ax4.set_title('test acc')
        ax4.plot(self.test_acc_history)
        plt.show();
        
        
    def confusion_matrix(self):
        actual = self.y_test
        predicted = self.y_test_pred_history[-1]
        labels = [0,1,2,3,4,5,6]

        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for a in list(zip(actual, predicted)):
            cm[a[0]][a[1]] += 1
            
        plt.figure(figsize=(7,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
        plt.yticks(rotation=0)
        plt.xlabel('Predicted Label', fontsize=10, labelpad=15)
        plt.ylabel('Actual Label', fontsize=10, labelpad=15)
        plt.title('confusion matrix', fontsize=12, pad=15)
        plt.show();
    

    def predict(self, X):
        
        o1 = X.dot(self.parameters['W1']) + self.parameters['b1']
        relu_out1 = self.relu(o1)
        o2 = relu_out1.dot(self.parameters['W2']) + self.parameters['b2']
        
        if self.hidden_size2 != 0:
            relu_out2 = self.relu(o2)
            scores = relu_out2.dot(self.parameters['W3']) + self.parameters['b3']
        else:
            scores = o2
            
        y_pred = np.argmax(scores, axis=1)

        return y_pred
        
        
