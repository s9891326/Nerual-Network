# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:44:07 2019

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.utils import np_utils # 用來後續將 label 標籤轉為 one-hot-encoding  


(X_train, y_train), (X_test, y_test) = mnist.load_data()

#try:
#    model = load_model("number_identify.h5")
#except:
print("new model")

model = Sequential()

model.add(Dense(units=256, input_dim=784, kernel_initializer="normal", activation="relu"))

model.add(Dense(units=10, kernel_initializer="normal", activation="softmax"))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# 將 training 的 label 進行 one-hot encoding，例如數字 7 經過 One-hot encoding 轉換後是 0000001000，即第7個值為 1
y_TrainOneHot = np_utils.to_categorical(y_train)
y_TestOneHot = np_utils.to_categorical(y_test)

# 將 training 的 input 資料轉為2維
X_train_2D = X_train.reshape(60000, 28*28).astype('float32')
X_test_2D = X_test.reshape(10000, 28*28).astype('float32')

X_Train_norm = X_train_2D/255
X_Test_norm = X_test_2D/255

#print("X_Train_norm = " , X_Train_norm)
#print("X_Test_norm = " , X_Test_norm)

# 進行訓練, 訓練過程會存在 train_history 變數中
train_history = model.fit(x=X_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=800, verbose=2)


# summarize layers
print(model.summary())

# 顯示訓練成果(分數)
scores = model.evaluate(X_Test_norm, y_TestOneHot)
print()
print("\t[info] acuracy of testinf data = {:2.1f}%".format(scores[1]*100.0))

# 預測(prediction)
X = X_Test_norm[0:10,:]
predictions = model.predict(X)

#print(predictions)
for i in range(0,predictions.shape[0]):
    pred = np.array(predictions[i])
    print("pred number = ", list(pred).index(predictions[i].max()))

plt.plot(train_history.history['loss'])  
plt.plot(train_history.history['val_loss'])  
plt.title('Train History')  
plt.ylabel('loss')  
plt.xlabel('Epoch')  
plt.legend(['loss', 'val_loss'], loc='upper left')  
plt.show() 

#save model
model.save("number_identify.h5")

