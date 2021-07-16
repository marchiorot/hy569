chest = 700
acc = 32
bvp = 64
eda = 4
temp = 4

def obtain_data(pkl_file, mode='binary'):
    wrist_label = []
    wrist_acc_x = []
    wrist_acc_y = []
    wrist_acc_z = []
    wrist_bvp = []
    wrist_eda = []
    wrist_temp = []
    for i in range(len(pkl_file['signal']['wrist']['EDA'])):
        wrist_label.append(pkl_file['label'][round(i*chest/eda)])
        wrist_acc_x.append(pkl_file['signal']['wrist']['ACC'][round(i*acc/eda)][0])
        wrist_acc_y.append(pkl_file['signal']['wrist']['ACC'][round(i*acc/eda)][1])
        wrist_acc_z.append(pkl_file['signal']['wrist']['ACC'][round(i*acc/eda)][2])
        wrist_bvp.append(pkl_file['signal']['wrist']['BVP'][round(i*bvp/eda)][0])
        wrist_eda.append(pkl_file['signal']['wrist']['EDA'][round(i*eda/eda)][0])
        wrist_temp.append(pkl_file['signal']['wrist']['TEMP'][round(i*temp/eda)][0])
    x = []
    y = []
    counter = 0
    for i in range(len(wrist_label)):
        if wrist_label[i] == 1 or wrist_label[i] == 2 or wrist_label[i] == 3 or wrist_label[i] == 4: 
            x.append([])
            x[counter].append(wrist_acc_x[i])
            x[counter].append(wrist_acc_y[i])
            x[counter].append(wrist_acc_z[i])
#             x[counter].append(wrist_bvp[i])
            x[counter].append(wrist_eda[i])
            x[counter].append(wrist_temp[i])
            y.append(wrist_label[i])
            counter +=1
    if mode == 'binary':
        Y = [0 if i==2 else 1 for i in y]
    elif mode == 'multiclass':
        Y = [[1 if y[i] == el+1 else 0 for el in range(4)] for i in range(len(y))]
    return(x,Y)
    

import pickle
import numpy as np
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import os
rootdir = 'C:/Users/Kapp/Desktop/UnCrete/569/datasets/WESAD/'

x_user_train = []
y_user_train = []
x_user_test = []
y_user_test = []
x_user_val = []
y_user_val = []

counter = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if '.pkl' in file:
            with open(subdir+'/'+file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            counter += 1
            x, Y = obtain_data(data, 'binary')
            if counter > 3:
                x_user_train.append(x)
                y_user_train.append(Y)
            else:
                x_user_test.append(x)
                y_user_test.append(Y)
                
                
x_train = [el for k in x_user_train for el in k]
y_train = [el for k in y_user_train for el in k]
x_test = [el for k in x_user_test for el in k]
y_test = [el for k in y_user_test for el in k]

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

model = Sequential()
model.add(Dense(100, input_dim=x_train.shape[1], activation='relu',
                kernel_initializer='random_normal'))
model.add(Dense(50,activation='relu',kernel_initializer='random_normal'))
model.add(Dense(25,activation='relu',kernel_initializer='random_normal'))
model.add(Dense(1,activation='sigmoid',kernel_initializer='random_normal'))
# model.add(Dense(y_train.shape[1],activation='softmax', kernel_initializer='random_normal'))
model.compile(loss='binary_crossentropy', 
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics =['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
    patience=5, verbose=1, mode='auto', restore_best_weights=True)

model.fit(x_train,y_train,validation_data=(x_test,y_test),
          callbacks=[monitor],verbose=2,epochs=1000)
          
model.save('model')