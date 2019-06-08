# -*- coding: utf-8 -*-
"""
MNIST預處理
"""
from keras.utils import np_utils
import numpy as np
np.random.seed(10)

#讀取資料
from keras.datasets import mnist
(x_train_image,y_train_label), (x_test_image,y_test_label)=mnist.load_data()
#將features使用reshape轉換數字5=>0,0,0,0,0,1,0,0,0,0,0
x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')
#將features標準化
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255
#將label以One-hot encoding轉換 ex
y_TrainOneHot =np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout

#建立模型
path = "D:\Spyder\MNIST\mlp_model_two_Dense.h5"

try:
    model = load_model(path)
    print("model = ", model)
except:
    model = Sequential()
    #建立輸入層和隱藏層
    model.add(Dense(units=1000,input_dim=784,kernel_initializer='normal',activation='relu'))
    """
        隱藏層=>256,輸入層=>784(28*28),使用常態分配的亂數初始化weight和bias,激活函數relu
        param = (input_dim + 1) * units
    """
    model.add(Dropout(0.5)) #加入dropout功能，避免overfitting
    
    #加入隱藏層2
    model.add(Dense(units=1000,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    
    #建立輸出層
    model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
    print(model.summary())
    """
        200960=784*256+256,2570=256*10+10
        Trainable params=200960+2570
        param = (input_dim + 1) * units
        input_dim = 上一層的units，上一層的輸出等於這層的輸入
    """
    
    """training"""
    #定義訓練方式
    #使用compile方法
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    """loss=>損失函數=>使用cross_entropy交叉熵 optimizer=>adam,metrics=>設定評估模型的方式是accuracy準確率"""
    #開始訓練
    train_history = model.fit(x=x_Train_normalize, y=y_TrainOneHot,validation_split=0.2,epochs=10,batch_size=200,verbose=2)
    """
    設定訓練與驗證資料比例0.2=>80%作為訓練資料=>60000*0.8,20%作為驗證資料=>60000*0.2,設定訓練週期=>10,每批次=>200筆
    設定顯示訓練過程:每次訓練週期顯示使用48000筆訓練,每批次200筆,所以分成240批次訓練,訓練完成後會計算在accuracy與loss,並且
    記錄在train_history
    """
    #儲存model
    model.save(path)



"""顯示訓練過程"""
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation): 
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation], color = 'red')
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'], loc='best') #設定圖例顯train,validation,位置在左上角
    plt.show()
show_train_history(train_history,'acc','val_acc') #藍色=>acc,橘色=>val_acc
show_train_history(train_history,'loss','val_loss')

"""以測試資料評估模型準確率"""
loss, acc = model.evaluate(x_Test_normalize, y_TestOneHot)
print()
print('accuracy=', acc)
"""進行預測"""
prediction=model.predict_classes(x_Test)

"""""images(數字影像) labels(真實值) prediction(預測結果) idx(開始顯示的資料index) num(要顯示的資料筆數，預設是10，不超過25)"""
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    """設定圖形大小"""
    if num>25: num=25 #如果顯示比數參數大於25設定為25，以免發生錯誤
    for i in range(0,num):
        ax=plt.subplot(5, 5, i + 1) #建立subgraph子圖形為5行5列
        ax.imshow(images[idx], cmap='binary') #畫出subgraph子圖形
        title= "label=" + str(labels[idx]) #設定子圖形title，顯示標籤欄位
        if len(prediction)>0:  #如果有傳入預測結果
            title += " predict=" + str(prediction[idx]) #標題title加入預測結果
            
        ax.set_title(title,fontsize=12) #設定子圖形的標題title與大小
        ax.set_xticks([]) #設定不顯示刻度
        ax.set_yticks([])
        idx+=1 #讀取下一筆
    plt.show() 
plot_images_labels_prediction(x_test_image,y_test_label,prediction,idx=1)

"""建立混淆矩陣，看那些最容易被混淆"""
import pandas as pd
matrix = pd.crosstab(y_test_label,prediction,colnames=['predict'],rownames=['label'])
print("matrix = ", matrix)

"""建立真實值與預測dataframe"""
df = pd.DataFrame({'label' : y_test_label, 'predict' : prediction})
#print("df = ", df)

print("----------------------")

for i in range(0, df.shape[0]):
    if (df.iloc[i, 0] != df.iloc[i, 1]):
        print(df[i:i+1])
#df[(df.label==5)&(df.predict==3)]

