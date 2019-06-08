# -*- coding: utf-8 -*-
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10) #設定隨機種子, 以便每次執行結果相同
(x_Train, y_Train),(x_Test, y_Test) = mnist.load_data()
"""將features轉為4為矩陣"""
x_Train4D = x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')
"""標準化"""
x_Train4D_normalized = x_Train4D / 255
x_Test4D_normalized = x_Test4D / 255
"""one-hot"""
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)
"""建立模型"""
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

path = "D:\Spyder\MNIST\cnn_minst.h5"

try:
    model = load_model(path)
    print("model = " , model)
except:
    model = Sequential()
    """建立捲積層1與池化層1"""
    model.add(Conv2D(filters=16,
                     kernel_size=(5,5),
                     padding='same',
                     input_shape=(28,28,1),
                     activation='relu'))
    """
        建立16個濾鏡，每個濾鏡大小5*5，讓卷積運算 產生的影像大小不變，輸入的影像大小28*28 因為是灰階影像所以維度是1
        邊界模式=same (填補=0, 步幅=1)
    """
    model.add(MaxPooling2D(pool_size=(2,2)))
    """縮減取樣，將28*28=>14*14"""
    
    
    """卷積層2與池化層2"""
    model.add(Conv2D(filters=16,kernel_size=(5,5),padding ='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) #14*14=>7*7
    model.add(Dropout(0.25)) #隨機放棄25%神經元避免overfitting
    """建立平坦層"""
    model.add(Flatten())
    """建立隱藏層"""
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    """建立輸出層"""
    model.add(Dense(10, activation = 'softmax'))
    print(model.summary())
    
    """進行訓練"""
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    train_history = model.fit(x=x_Train4D_normalized,y=y_TrainOneHot,validation_split=0.2,epochs=10,batch_size=300,verbose=2)
    """準確率"""
    scores = model.evaluate(x_Test4D_normalized,y_TestOneHot)
    print()
    print('accuracy=',scores[1])
    
    # =============================================================================
    # 儲存model
    model.save(path)
    # =============================================================================

"""執行預測"""
prediction = model.predict_classes(x_Test4D_normalized)
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    """設定圖形大小"""
    if num>25: num=25 #如果顯示比數參數大於25設定為25，以免發生錯誤
    for i in range(0,num):
        ax=plt.subplot(5,5, 1+i) #建立subgraph子圖形為5行5列
        ax.imshow(images[idx], cmap='binary') #畫出subgraph子圖形
        title= "label=" +str(labels[idx]) #設定子圖形title，顯示標籤欄位
        if len(prediction)>0:  #如果有傳入預測結果
            title+="predict=" +str(prediction[idx]) #標題title加入預測結果
            
        ax.set_title(title,fontsize=10) #設定子圖形的標題title與大小
        ax.set_xticks([]);ax.set_yticks([]) #設定不顯示刻度
        idx+=1 #讀取下一筆
    plt.show() 
plot_images_labels_prediction(x_Test,y_Test,prediction,idx=0)