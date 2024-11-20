import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mean_squared_error(Y_true, Y_pred):
    return np.sum(np.square(Y_true-Y_pred))  * 1/Y_true.shape[0] 
def fit_linear_regression(X, y,learning_rate):
    X = np.append(X,np.ones((X.shape[0],1)),1)
    weights = np.zeros(X.shape[1])
    prev_error = -1
    epoch = 1
    i = 0
    while True:
        i+=1
        if i == 100000 : break
        pred = X.dot(weights)
        update = np.matmul(np.transpose(pred - y) * 1/X.shape[0],X) * learning_rate 
        weights = weights - update
        print(f"epoch {epoch}: loss = {mean_squared_error(y,pred)}")
        if(prev_error != -1 and prev_error < mean_squared_error(y,pred)): break
        prev_error = mean_squared_error(y,pred)
        epoch+=1
    return weights

def predict(X, weights):
    X = np.append(X,np.ones((X.shape[0],1)),1)
    return X.dot(weights)

def plot_preds(X, Y_true, Y_pred):

    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    Y_true_sorted = Y_true[sort_idx]
    Y_pred_sorted = Y_pred[sort_idx]

    plt.plot(X_sorted,Y_true_sorted,color = 'g')
    plt.plot(X_sorted,Y_pred_sorted,color = 'r')
    plt.show()
def normalize_features(X):
    sum = np.sum(X,axis=0)
    sum = sum / X.shape[0]
    X = X - sum
    standard_deviation = np.sqrt(np.sum(np.square(X),axis=0) / X.shape[0])
    return X / standard_deviation
    

if __name__ == "__main__":
    # Load the train data
    train_data = pd.read_csv('train.csv')
    
    X_train = train_data[['carlength', 'carwidth', 'carheight', 'horsepower', 'peakrpm']].values
    y_train = train_data['price'].values

    test_data = pd.read_csv('test.csv')
    
    X_test = test_data[['carlength', 'carwidth', 'carheight', 'horsepower', 'peakrpm']].values
    y_test = test_data['price'].values
    

    ############# Without normailzed features ################
    weights = fit_linear_regression(X_train, y_train,1e-9 * 7.5)
    
    y_pred = predict(X_test, weights)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse}")
    
    plot_preds(test_data['horsepower'].values, y_test, y_pred)
    
    ################### Normalize features ######################
    X_train = normalize_features(X_train)
    
    weights = fit_linear_regression(X_train, y_train,1e-3)
    X_test = normalize_features(X_test)
    
    y_pred = predict(X_test, weights)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse}")
    
    plot_preds(test_data['horsepower'].values, y_test, y_pred)