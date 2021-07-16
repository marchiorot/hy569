import numpy as np
import tensorflow as tf
from tensorflow import keras
from functools import reduce


model = keras.models.load_model('model')

import time

while True:
    
    starttime = time.time()
    with open('../e4_server/e4_ACC/bin/Debug/netcoreapp3.1/acc.txt') as f:
        acc = f.readlines()
    with open('../e4_server/e4_gsr/bin/Debug/netcoreapp3.1/gsr.txt') as f:
        eda = f.readlines()
    with open('../e4_server/ble-client-windows-master/bin/Debug/tmp.txt') as f:
        tmp = f.readlines()
    acc_x_last = []
    acc_y_last = []
    acc_z_last = []
    eda_last = []
    tmp_last = []
    for i in ([-j-1 for j in range(50)]):
    # for i in ([-1,-2,-3,-4,-5]):
        acc_x_last.append(float(acc[i].strip().split(' ')[-3]))
        acc_y_last.append(float(acc[i].strip().split(' ')[-2]))
        acc_z_last.append(float(acc[i].strip().split(' ')[-1]))
        eda_last.append(float(eda[i].strip().split(' ')[-1]))
        tmp_last.append(float(tmp[i].strip().split(' ')[-1]))
    to_predict = [[acc_x_last[i] , acc_y_last[i], acc_z_last[i], eda_last[i], tmp_last[i]] for i in range(len(acc_x_last))]

    pred = model.predict(to_predict)
    predicts = np.round(pred)
    unique, counts = np.unique(predicts, return_counts=True)
    final_values = dict(zip(unique, counts))
    if 1.0 not in final_values.keys():
        final_values[1.0] = 0
    if 0.0 not in final_values.keys():
        final_values[0.0] = 0

    probabilities = [i if i > 0.5 else i-1 for i in pred]
    final_prob = abs(reduce(lambda x,y: x+y ,probabilities)/len(probabilities))[0]
    confidence = final_prob
    
    # print(pred) 
    
    stress = ''
    if final_values[0.0] > final_values[1.0]:
        stress = 'stressed'
    else:
        stress = 'not stressed'
    
    with open('stress.txt', 'w') as f:
        f.write(stress)
    
    with open('probs.txt', 'w') as f:
        f.write(str(confidence))
       
    # print(confidence)

    time.sleep(30.0 - ((time.time() - starttime) % 30.0))