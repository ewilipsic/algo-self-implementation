import numpy as np

y_pred = np.array([1,1,1,0,1])
y_true = np.array([1,0,0,0,1])


TP = 0
FP = 0
TN = 0
FN = 0
flag = 0
for i in range(y_pred.shape[0]):
    
    if(y_pred[i] == 0 and y_true[i] == 0):
        TN += 1
        flag+=1.1
    if(y_pred[i] == 0 and y_true[i] == 1):
        FN += 1
        flag+=1.01
    if(y_pred[i] == 1 and y_true[i] == 0):
        FP += 1
        flag+=1.001
    if(y_pred[i] == 1 and y_true[i] == 1):
        TP += 1
        flag+=1.0001
    if(flag>2):
        print('flag ',flag)
        print(y_pred[i])
        print(y_true[i])
    flag = 0
    print()