
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np  
#import itertools
import os
import cv2

#使用tf.keras.applications中的CNN各種訓練模型及預載權重
def create_model(model_no, loaded_imagenet, freeze_layers): 
    if loaded_imagenet == 'Y':
        if model_no == 1:
            covn_base = tf.keras.applications.DenseNet201(weights='imagenet',include_top=False,input_shape=(224,224,3))
        elif model_no == 2:
            covn_base = tf.keras.applications.ResNet50V2(weights='imagenet',include_top=False,input_shape=(224,224,3))
        elif model_no == 3:
            covn_base = tf.keras.applications.InceptionV3(weights='imagenet',include_top=False,input_shape=(224,224,3))
    elif loaded_imagenet == 'N':
        if model_no == 1:
            covn_base = tf.keras.applications.DenseNet201(weights=None,include_top=False,input_shape=(224,224,3))
        elif model_no == 2:
            covn_base = tf.keras.applications.ResNet50V2(weights=None,include_top=False,input_shape=(224,224,3))
        elif model_no == 3:
            covn_base = tf.keras.applications.InceptionV3(weights=None,include_top=False,input_shape=(224,224,3))   

    if freeze_layers == 0:
        covn_base.trainable = True
    else:
        covn_base.trainable = True
        for layers in covn_base.layers[:freeze_layers]:
            layers.trainable = False
    return covn_base

#4GB GPU RAM 依各模型及預載imagenet凍結層數後情況設定batch_size，Densenet非常吃記憶體
def get_batch_size(model_no,loaded_imagenet):
    if loaded_imagenet == 'N':
        if model_no == 1:
            batch_size = 8
        else:
            batch_size = 32
    elif loaded_imagenet == 'Y':
        batch_size = 64
    return batch_size

#使用的模型參數設定
#模型編號 {1:'DenseNet201', 2:'ResNet50V2', 3:'InceptionV3'}
model_no = 1

#是否預載入imagenet權重 Y/N
loaded_imagenet = 'N'

#是否載入自已訓練過的權重 Y/N，選Y需輸入檔名，載入的權重檔需和目前要訓練的模型結再參數一模一樣(訓練層數及凍結層數)
load_weight = 'N'
load_weight_file = 'weight_chest_xray_PNEUMONIA_DenseNet201_N_100.h5'

#凍結訓練層數負數為只訓練最後層數，正數為凍結層數
#預載imagenet權重時只訓練最後5層，不預載時不凍結
if loaded_imagenet == 'Y':
    freeze_layers = -5
elif loaded_imagenet == 'N':
    freeze_layers = 0

#設置圖片尺寸
im_height = 224
im_width = 224

#一次選取的樣本數，可依記憶體大小調整
#batch_size = get_batch_size(model_no,loaded_imagenet)
batch_size = 8
#迭代次數
epochs = 100

#資料集路徑
image_path = 'chest_xray721'
train_dir = './data_set/{}/train'.format(image_path) 
validation_dir = './data_set/{}/val'.format(image_path)

#分類數
class_num = len(os.listdir(train_dir))

# 建立各種儲存路徑目錄
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")
if not os.path.exists("model"):
    os.makedirs("model")
if not os.path.exists("loss"):
    os.makedirs("loss")
if not os.path.exists("predict"):
    os.makedirs("predict")

#設置各種存取檔案的路徑及檔名
model_dict = {1:'DenseNet201', 2:'ResNet50V2', 3:'InceptionV3'}
model_name = model_dict[model_no]
save_file_name = '{}_{}_{}_{}'.format(image_path,model_name,loaded_imagenet,epochs)
save_weight_name = './save_weights/weight_{}.h5'.format(save_file_name)
save_model_name = './model/model_{}.h5'.format(save_file_name)
save_plt_name = './loss/loss_{}.png'.format(save_file_name)
save_predict_name = './predict/predict_{}.jpg'.format(save_file_name)

#有預載權重時組合出路徑，無預找時路徑為空
if load_weight == 'Y':
    load_weight_name = './save_weights/{}'.format(load_weight_file)
elif load_weight == 'N':
    load_weight_name = ''


#定義訓練圖片生成器
train_image_generator = ImageDataGenerator( rescale=1./255, #歸一化 
                                             rotation_range=40,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True,
                                             fill_mode='nearest')
                                            
#讀取訓練圖片及lable
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width), #resize圖片到224x224
                                                           class_mode='categorical') #one-hot編碼
                                                           
#訓練圖片數量        
total_train = train_data_gen.n 

#定義驗証圖片生成器
validation_image_generator = ImageDataGenerator(rescale=1./255)

#讀取驗証圖片及lable
val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=False,
                                                              target_size=(im_height, im_width),
                                                              class_mode='categorical')
                                                              
#驗証圖片數量      
total_val = val_data_gen.n


#建立模型
covn_base = create_model(model_no,loaded_imagenet,freeze_layers)        
model = tf.keras.Sequential()
model.add(covn_base)
model.add(tf.keras.layers.GlobalAveragePooling2D())  #添加全局平均池化層
model.add(tf.keras.layers.Dense(1024,activation='relu'))  #添加全連接層
model.add(tf.keras.layers.Dropout(rate=0.5))  #添加Dropout層，防止過擬合
model.add(tf.keras.layers.Dense(class_num,activation='softmax'))  #添加輸出層(分類數)
#載入之前訓練的權重
if load_weight_name: 
    print('loaded weight = {}'.format(load_weight_name))
    model.load_weights(load_weight_name,by_name=True) 
model.summary()   #印出每層參數表

#編譯模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), #使用adam優化器，學習率為0.0001
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), #交叉熵損失函數
              metrics=["accuracy"]) #評價函數

#回调函数1:学习率衰减
reduce_lr = ReduceLROnPlateau(
                                monitor='val_loss', #需要监视的值
                                factor=0.1,  #学习率衰减为原来的1/10
                                patience=2,  #当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
                                mode='auto', #当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min，在auto模式下，评价准则由被监测值的名字自动推断
                                verbose=1 #如果为True，则为每次更新输出一条消息，默认值:False
                             )
#回调函数2:保存最优模型
checkpoint = ModelCheckpoint(
                                filepath='./save_weights/myDenseNet201.ckpt', #保存模型的路径
                                monitor='val_loss', #需要监视的值
                                save_weights_only=False, #若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
                                save_best_only=True, #当设置为True时，监测值有改进时才会保存当前的模型
                                mode='auto', #当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min，在auto模式下，评价准则由被监测值的名字自动推断
                                save_freq=1 #CheckPoint之间的间隔的epoch数
                            )

#開始訓練
history = model.fit(x=train_data_gen,   #輸入生成的訓練集
                    steps_per_epoch=total_train // batch_size,
                    epochs=epochs, #迭代次數
                    validation_data=val_data_gen,  #輸入生成的驗証集
                    validation_steps=total_val // batch_size,
                    callbacks=[reduce_lr]) #执行回调函数
                    
#儲存訓練好的權重及模型                    
model.save_weights(save_weight_name) 
model.save(save_model_name)
#記錄訓練集和驗証集的準確率
history_dict = history.history
train_loss = history_dict["loss"]
train_accuracy = history_dict["accuracy"]
val_loss = history_dict["val_loss"]
val_accuracy = history_dict["val_accuracy"]

#畫出準確率和loss值
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(range(epochs), train_accuracy, label='train_accuracy')
plt.plot(range(epochs), val_accuracy, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.suptitle(save_model_name.split('/')[-1],fontsize=20) #定義表格名稱
plt.savefig(save_plt_name) #儲存圖片
plt.show()


#預測訓練集及驗証集中的圖片各兩張，測試模型是否正常
image_list = []
class_dirs = os.listdir(train_dir)
for class_dir in class_dirs:
    file_path = '{}/{}'.format(train_dir,class_dir)
    file_list = []
    for root, dirs, files in os.walk(file_path):
        if files:
            for f in files:
                file_temp = '{}/{}'.format(root,f)
                file_temp = file_temp.replace('\\','/')
                file_list.append(file_temp)
    image_list.append(file_list[0])

#訯定分類碼
inverse_dict = dict(enumerate(class_dirs))

#載入圖片並處理
predict_list = []
i = 1
for image_file in image_list:
    img = cv2.imread(image_file)
    img2 = img.copy()
    img2 = cv2.resize(img2,(224,224))
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    img2 = img2 /255
    #預測
    img2 = (np.expand_dims(img2, 0))
    result = np.squeeze(model.predict(img2))
    predict_list.append(result)
    predict_class = np.argmax(result)
    #預測結果印在圖片上
    txt = str(int(predict_class)) + ' ' + str(round((result[predict_class]*100),1)) + '%'
    txt2 = str(inverse_dict[int(predict_class)])
    print(txt,txt2)
    img3 = cv2.resize(img,(224,224))
    img3 = cv2.putText(img3,txt,(10,30),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,0),2)
    img3 = cv2.putText(img3,txt2,(10,60),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,0),2)
    #合成一張圖片
    if i == 1:
        image_stack = img3
    else:
        image_stack = np.hstack((image_stack,img3))
    i = i + 1
#儲存圖片並顯示
cv2.imwrite('{}'.format(save_predict_name),image_stack)    
cv2.imshow('image_stack',image_stack)
cv2.waitKey(0)
cv2.destroyAllWindows()

