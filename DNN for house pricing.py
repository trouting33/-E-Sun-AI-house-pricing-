import pandas as pd
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import h5py
import keras
import tensorflow as tf
from keras import metrics
from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model



file1 = "train.csv"                                        #train.csv，此例中不使用
data1 = pd.read_csv(file1)

file2 = "train_houser.csv"                                 #train_houser.csv為特定篩選過的欄位
data2 = pd.read_csv(file2)                               

data11=pd.get_dummies(data1['village'])                    #處理train.csv的資料，但此例中不使用

frames = [data11,data2]

data_combined = pd.concat(frames, axis=1)                  #處理train.csv的資料，但此例中不使用

#############################################資料載入分隔線###################################################

data = data2                                               #使用train_houser.csv
label_col = 'total_price'                                  #將'total_price'字串特別存起來，因為它是output Y



''' THE FOLLOWING CODE IS SAMPLE FROM KAGGLE
data_csv['sale_yr'] = pd.to_numeric(data_csv.date.str.slice(0, 4))
data_csv['sale_month'] = pd.to_numeric(data_csv.date.str.slice(4, 6))
data_csv['sale_day'] = pd.to_numeric(data_csv.date.str.slice(6, 8))

data = pd.DataFrame(data_csv, columns=[
        'sale_yr','sale_month','sale_day',
        'bedrooms','bathrooms','sqft_living','sqft_lot','floors',
        'condition','grade','sqft_above','sqft_basement','yr_built',
        'zipcode','lat','long','sqft_living15','sqft_lot15','price'])
label_col = 'price'
'''

def train_validate_test_split(df, train_part=0.9, validate_part=0.05, test_part=0.05, seed=None):    #資料切割函數
	np.random.seed(seed)
	total_size = train_part + validate_part + test_part
	train_percent = train_part / total_size
	validate_percent = validate_part / total_size
	test_percent = test_part / total_size
	perm = np.random.permutation(df.index)
	m = len(df)
	train_end=int(train_percent * m)
	validate_end = int(validate_percent * m) + train_end
	train = perm[:train_end]
	validate = perm[train_end:validate_end]
	test = perm[validate_end:]
	return train, validate, test
	
train_size, valid_size, test_size = (80, 20, 0)                                                      #設定訓練、驗證的比例
data_train, data_valid, data_test = train_validate_test_split(data,                                  #call上面資料切割的def，回傳被切割好的資料
                              train_part=train_size, 
                              validate_part=valid_size,
                              test_part=test_size,
                              seed=2017)
							  
data_y_train = data.loc[data_train, [label_col]]                                                     #設定訓練資料的output Y('total_price')
data_x_train = data.loc[data_train, :].drop(label_col, axis=1)                                       #設定訓練資料的input X('total_price'外的所有資料)
data_y_valid = data.loc[data_valid, [label_col]]                                                     #設定驗證資料的output Y('total_price')
data_x_valid = data.loc[data_valid, :].drop(label_col, axis=1)                                       #設定驗證資料的input X('total_price'外的所有資料)

print('Size of training set: ', len(data_x_train))                                                   #印出上面相關資料
print('Size of validation set: ', len(data_x_valid))
print('Size of test set: ', len(data_test), '(not converted)')

def norm_stats(df1, df2):                                                                            #算每個欄位的minimum, maximum, mu, sigma的函數
    dfs = df1.append(df2)                                                                            #             極小值  極大值  平均 標準差
    minimum = np.min(dfs)
    maximum = np.max(dfs)
    mu = np.mean(dfs)
    sigma = np.std(dfs)
    return (minimum, maximum, mu, sigma)
	
def z_score(col, stats):                                                                             #算Z分數(標準化)的函數 
    epsilon = 0.00000001                                                                             #註:公式分母加上極小值(epsilon)為避免除以0
    m, M, mu, s = stats
    df = pd.DataFrame()
    for c in col.columns:
        df[c] = (col[c]-mu[c])/(s[c] + epsilon)                                                      #分母加epsilon避免為0
    return df
	
stats = norm_stats(data_x_train, data_x_valid)                                                       #call函數計算並回傳minimum, maximum, mu, sigma
arr_x_train = np.array(z_score(data_x_train, stats))                                                 #call函數計算訓練集欄位標準化值(output Y以外)
arr_y_train = np.array(data_y_train)
arr_x_valid = np.array(z_score(data_x_valid, stats))                                                 #call函數計算驗證集欄位標準化值(output Y以外)
arr_y_valid = np.array(data_y_valid)

print('Training shape:', arr_x_train.shape)                                                          #印出上面相關資料
print('Training samples: ', arr_x_train.shape[0])
print('Validation samples: ', arr_x_valid.shape[0])


def basic_model_3(x_size, y_size):                                                                   #建立模型的函數
    t_model = Sequential()                                                                           #Sequential代表一層一層疊出神經網路
    t_model.add(Dense(500, activation="tanh", kernel_initializer='normal', input_shape=(x_size,)))    #.add代表增加一層(我設定80個節點、tanh函數)
    t_model.add(Dropout(0.2))                                                                        #啟動Dropout功能(避免過度依賴特定節點，會過度配適)
    t_model.add(Dense(600, activation="relu", kernel_initializer='normal', 
        kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))

#    t_model.add(Dropout(0.1))
#    t_model.add(Dense(60, activation="relu", kernel_initializer='normal', 
#        kernel_regularizer=regularizers.l1_l2(0.01), bias_regularizer=regularizers.l1_l2(0.01)))

		
#    t_model.add(Dropout(0.2))
#    t_model.add(Dense(60, activation="relu", kernel_initializer='normal', 
#        kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    t_model.add(Dropout(0))
    t_model.add(Dense(300, activation="relu", kernel_initializer='normal', 
        kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))		
    t_model.add(Dropout(0))
    t_model.add(Dense(300, activation="relu", kernel_initializer='normal', 
        kernel_regularizer=regularizers.l1_l2(0.01), bias_regularizer=regularizers.l1_l2(0.01)))

    t_model.add(Dropout(0))
    t_model.add(Dense(70, activation="relu", kernel_initializer='normal'))
    t_model.add(Dropout(0))
    t_model.add(Dense(y_size))                                                                       #設定output數量(玉山60000筆，我們就塞60000)
    nadam = optimizers.Nadam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    t_model.compile(                                                                                 #編譯模型，生成loss function 參照optimizer
        loss='mean_absolute_error',                                                                  
        optimizer=nadam,
        metrics=[metrics.mae])
    return(t_model)
print(arr_x_train.shape)
print(arr_y_train.shape)
print(arr_x_train.shape[1])
print(arr_y_train.shape[1])
model = basic_model_3(arr_x_train.shape[1], arr_y_train.shape[1])                                    #呼叫上面model函數模型
model.summary()

epochs = 120                                                                                         #設定學習128輪(經驗決定)
batch_size = 128                                                                                     #設定批量學習，每批128筆資料(經驗決定)

print('Epochs: ', epochs)
print('Batch size: ', batch_size)                                                                    #印出上面epoch、batch size資料

history = model.fit(arr_x_train, arr_y_train,                                                        #將訓練結果和output Y做比較，存在history內
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=2, # Change it to 2, if wished to observe execution
    validation_data=(arr_x_valid, arr_y_valid),
    callbacks=None)
	
train_score = model.evaluate(arr_x_train, arr_y_train, verbose=0)                                    #評估並讀取訓練結果，以每筆的MAE、loss表示
valid_score = model.evaluate(arr_x_valid, arr_y_valid, verbose=0)                                    #評估並讀取驗證結果，以每筆的MAE、loss表示
                                                                                                                              #平均誤差 每筆誤差函數
print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4)) 
print('Val MAE: ', round(valid_score[1], 4), ', Val Loss: ', round(valid_score[0], 4))

def plot_hist(h, xsize=6, ysize=10):                                                                 #畫出上面的成果的函數
    # Prepare plotting
    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [xsize, ysize]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)
    
    # summarize history for MAE
    plt.subplot(211)
    plt.plot(h['mean_absolute_error'])
    plt.plot(h['val_mean_absolute_error'])
    plt.title('Training vs Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # summarize history for loss
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot it all in IPython (non-interactive)
    plt.draw()
    plt.show()

    return
	
plot_hist(history.history, xsize=8, ysize=12)                                                        #CALL上面的函數，畫出逐輪訓練結果
#########################################################################

model.save("train.h5")                                                                               #儲存訓練模型，於此程式資料夾中產生"train.h5"檔案
del model

FileTest = "testr.csv"                                                                               #匯入測試模型
datatest = pd.read_csv(FileTest)

def test_norm_stats(df):                                                                             #和上面建立模型一樣，標準化流程
    minimum = np.min(df)
    maximum = np.max(df)
    mu = np.mean(df)
    sigma = np.std(df)
    return (minimum, maximum, mu, sigma)

def test_z_score(col, stats):                                                                        #和上面建立模型一樣，標準化流程
    epsilon = 0.00000001
    m, M, mu, s = stats
    df = pd.DataFrame()
    for c in col.columns:
        df[c] = (col[c]-mu[c])/(s[c] + epsilon)                                                      #分母加epsilon避免為0
    return df

stats = test_norm_stats(datatest)
arr_x_test = np.array(test_z_score(datatest, stats))                                                 #和上面建立模型一樣，標準化流程
	
	
new_model = keras.models.load_model('train.h5')                                                      #導入模型
predictions = new_model.predict(arr_x_test)                                                          #將測試集套入模型，取得結果
print(predictions)                                                                                   #印出預測結果

#print(np.argmax(predictions[0]))

with open('output2.csv', 'w', newline='') as csvfile:                                                #將預測結果寫入output2.csv
	writer = csv.writer(csvfile)
	for p in predictions:
		writer.writerow(p)


