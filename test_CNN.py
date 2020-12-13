
import tensorflow as tf
import numpy as np  
import cv2
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

#載入模型
model_file = 'model_chest_xray721_DenseNet201_N_100.h5'
model_path = './model/'
model = tf.keras.models.load_model('{}/{}'.format(model_path,model_file))
#預測用圖片集 
test_dir = 'chest_xray721'
test_path = './data_set/{}/test'.format(test_dir)
test_files = []
test_files = [os.path.join(root,name) for root,dirs,files in os.walk(test_path) for name in files]
test_files = [test_file.replace('\\','/') for test_file in test_files]  
#抽取test圖片
sample_num = ''
if sample_num:
    if int(sample_num) > len(test_files):
        sample_num = len(test_files)
    image_list = random.sample(test_files,int(sample_num))
else: 
    image_list = random.sample(test_files,len(test_files))
print('Test samples {} files'.format(len(image_list)))

#設定分類碼
class_dirs = os.listdir(test_path)
labels_dict = dict(enumerate(class_dirs))
inverse_labels = dict(zip(labels_dict.values(),labels_dict.keys()))
#ROC positive label
roc_label = 1

#載入圖片並處理
predict_list = []
i = 1
for image_file in image_list:
    img = cv2.imread(image_file)
    if img is None:
        print(" Can not read {}".format(image_file))
        continue
    img2 = img.copy()
    img2 = cv2.resize(img2,(224,224))
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    img2 = img2 /255
    #預測
    img2 = (np.expand_dims(img2, 0))
    result = np.squeeze(model.predict(img2))

    class_name = image_file.split('/')[-2]
    label = inverse_labels[class_name]

    predict_class = np.argmax(result)
    if predict_class == label:
        point = 1
    else: 
        point = 0
    predict_list.append([image_file.split('/')[-1],image_file.split('/')[-2],label,predict_class,point,round(result[predict_class]*100,1),result[roc_label]])
    #預測結果印在圖片上
    txt = str(labels_dict[int(predict_class)]) + ' ' +str(round((result[predict_class]*100),1)) + '%'
    
    #print('\r',image_file.split('/')[-1],txt,end='')
    print('\r{}/{}'.format(i,len(image_list)),end='')
    i = i + 1
df = pd.DataFrame(predict_list)
df.columns = ['image','dir','label','predict','point','predict_rate','y_pred_keras']

fpr_keras, tpr_keras, thresholds_keras = roc_curve(df['label'].to_list(), df['y_pred_keras'].to_list(),pos_label=roc_label)
auc_keras = auc(fpr_keras, tpr_keras)

score = round(df['point'].sum() / df['point'].count(),2)
print('\nScore = {}% {}/{} \n{} AUC = {}'.format(score*100, df['point'].sum(), df['point'].count(),labels_dict[roc_label],round(auc_keras,3)))

if not os.path.exists('./report'):
    os.makedirs('./report')
df.to_csv('./report/{}_{}.csv'.format(model_file,int(score*100)))


plt.figure(2)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='{} (area = {:.3f})'.format(labels_dict[roc_label],auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('{}\nROC curve'.format(model_file))
plt.legend(loc='best')
if not os.path.exists("ROC"):
    os.makedirs("ROC")
plt.savefig('./ROC/{}_roc.png'.format(model_file.split('.')[-2]))
#plt.show()

def plot_confusion_matrix(cm, target_names,model_file,cmap=None):
    accuracy = np.trace(cm) / float(np.sum(cm)) #計算準確率
    misclass = 1 - accuracy #計算錯誤率
    if cmap is None:
        cmap = plt.get_cmap('Blues') #顏色設置成藍色
    plt.figure(figsize=(9, 8)) #設置視窗尺寸
    plt.imshow(cm, interpolation='nearest', cmap=cmap) #顯示圖片
    plt.title('{}\nConfusion matrix'.format(model_file)) #顯示標題
    plt.colorbar() #繪製顏色條

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names) #x座標
        plt.yticks(tick_marks, target_names, rotation=90) #y座標標籤旋轉90度

    pm = cm.astype('float32') / cm.sum(axis=1)
    pm = np.round(pm,2) #對數字保留兩位元小數

    thresh = cm.max() / 1.5 
    #if normalize else cm.max() / 2
    #將cm.shape[0]、cm.shape[1]中的元素組成元組，遍歷元組中每一個數字
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 

        plt.text(j, i, "{:,} ({:0.2f})".format(cm[i, j],pm[i, j]),
                    horizontalalignment="center",  #數字在方框中間
                    color="white" if cm[i, j] > thresh else "black") #設置字體顏色

    plt.tight_layout() #自動調整子圖參數,使之填充整個圖像區域
    plt.subplots_adjust(left = 0.08, bottom = 0.09)
    plt.ylabel('True label') #y方向上的標籤
    plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass)) #x方向上的標籤
    #儲存圖片
    if not os.path.exists("confusion_matrix"):
        os.makedirs("confusion_matrix")
    plt.savefig('./confusion_matrix/{}_cm.png'.format(model_file.split('.')[-2]))
    plt.show() #顯示圖片

#標籤，存入到labels中
#labels = [labels_dict[0],labels_dict[1]]

# 預測驗證集資料整體準確率
#Y_pred = model.predict_generator(val_data_gen, total_val // batch_size + 1)
# 將預測的結果轉化為one hit向量
#Y_pred_classes = np.argmax(Y_pred, axis = 1)
# 計算混淆矩陣
confusion_mtx = confusion_matrix(y_true = df['label'].to_list(),y_pred = df['predict'].to_list())
# 繪製混淆矩陣
plot_confusion_matrix(confusion_mtx, target_names=class_dirs, model_file=model_file)
