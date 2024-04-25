from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter import simpledialog
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import cv2
import os
import numpy as np

#loading python require packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import pickle
from keras.callbacks import ModelCheckpoint
import pickle

main = tkinter.Tk()
main.title("Pancreatic Tumor Detection using Image Processing") #designing main screen
main.geometry("1000x650")


global filename, images, mask, cnn_model
global train_vol, test_vol, train_seg, test_seg

def loadDataset():
    global images, mask
    if os.path.exists("model/X.npy"):
        images = np.load("model/X.npy")
        mask = np.load("model/Y.npy")        
    else:
        for root, dirs, directory in os.walk("Dataset/images"):
            for j in range(len(directory)):
                name = directory[j]
                if os.path.exists("Dataset/mask/"+name):
                    img = cv2.imread("Dataset/images/"+directory[j],0)
                    img = cv2.resize(img,(128, 128), interpolation = cv2.INTER_CUBIC)
                    X.append(img)
                    img = cv2.imread("Dataset/mask/"+name,0)
                    img = cv2.resize(img,(128, 128), interpolation = cv2.INTER_CUBIC)
                    Y.append(img)
                    print(name+" "+directory[j])
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save("model/X",X)
        np.save("model/Y",Y)

def uploadDataset():
    global filename, images, mask
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    images = []
    mask = []
    loadDataset()
    text.insert(END,"Total images loaded = "+str(images.shape[0]))

def processDataset():
    global images, mask
    text.delete('1.0', END)
    dim = 128
    img = images[0]
    images = np.reshape(images,(images.shape[0], dim,dim,1))
    mask = np.reshape(mask,(mask.shape[0], dim,dim,1))
    text.insert(END,"Dataset Processing & Normalization Complated")
    img = cv2.resize(img, (300, 300))
    cv2.putText(img, "Sample Processed Image", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    cv2.imshow('Processed Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def trainTestSplit():
    global images, mask
    global train_vol, test_vol, train_seg, test_seg
    text.delete('1.0', END)
    train_vol, validation_vol, train_seg, validation_seg = train_test_split((images-127.0)/127.0, (mask>127).astype(np.float32), test_size = 0.1,random_state = 2018)
    train_vol, test_vol, train_seg, test_seg = train_test_split(train_vol,train_seg, test_size = 0.2, random_state = 2018)
    text.insert(END,"Dataset Training & Testing Details\n\n")
    text.insert(END,"80% images for training : "+str(train_vol.shape[0])+"\n")
    text.insert(END,"20% images for testing  : "+str(test_vol.shape[0])+"\n")

#function to calculate all variants of unet algorithms on test images
def calculateMetrics(cnn_model_type):
    lists = np.empty([1,128,128,1])
    test = 'Dataset/images/334.png'
    img = cv2.imread(test,0)
    img = cv2.resize(img,(128,128), interpolation = cv2.INTER_CUBIC)
    img = img.reshape(1,128,128,1)
    img = (img-127.0)/127.0
    preds = cnn_model_type.predict(img)#predict segmented image
    preds = preds[0]
    cv2.imwrite("test.png",preds*255)
    x = cv2.imread("test.png",0)
    mask = cv2.imread('Dataset/mask/334.png',0)
    mask = cv2.resize(mask,(128,128), interpolation = cv2.INTER_CUBIC)
    FP = len(np.where(x - mask  == -1)[0])
    FN = len(np.where(x - mask  == 1)[0])
    TN = len(np.where(x + mask == 2)[0])
    TP = len(np.where(x + mask == 0)[0])
    if FN == 0:
        FN = 1
    if TN == 0:
        TN = 1
    print(str(FP)+" "+str(FN)+" "+str(TN)+" "+str(TP))
    accuracy = (TP + TN) / (TP+TN+FP+FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fscore = (2 * precision * recall) / (precision + recall)
    text.insert(END,"CNN Segmentation Accuracy : "+str(accuracy)+"\n")
    

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)   

def getCNNModel(input_size=(128,128,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv1) #adding dilation rate for all layers
    conv1 = Dropout(0.1) (conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
 
    conv2 = Conv2D(64, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv2)
    conv2 = Dropout(0.1) (conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
 
    conv3 = Conv2D(128, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool2)#adding dilation to all layers
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), dilation_rate=2, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), dilation_rate=2, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), dilation_rate=2, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), dilation_rate=2, activation='relu', padding='same')(up9)#adding dilation
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)#not adding dilation to last layer

    return Model(inputs=[inputs], outputs=[conv10])               

def runCNN():
    text.delete('1.0', END)
    global cnn_model
    global train_vol, test_vol, train_seg, test_seg
    cnn_model = getCNNModel(input_size=(128, 128, 1))
    cnn_model.compile(optimizer=Adam(learning_rate=1e-4), loss=[dice_coef_loss], metrics = [dice_coef, 'binary_accuracy']) #compiling model
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(x = train_vol, y = train_seg, batch_size = 64, epochs = 20, validation_data=(test_vol,test_seg), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    calculateMetrics(cnn_model)

def values(filename, acc, loss):
    f = open(filename, 'rb')
    train_values = pickle.load(f)
    f.close()
    accuracy_value = train_values[acc]
    loss_value = train_values[loss]
    return accuracy_value, loss_value

def graph():
    train_acc, train_loss = values("model/cnn_history.pckl", "binary_accuracy", "loss")
    val_acc, val_loss = values("model/cnn_history.pckl", "val_binary_accuracy", "val_loss")

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(train_acc, 'ro-', color = 'green')
    plt.plot(train_loss, 'ro-', color = 'blue')
    plt.plot(val_acc, 'ro-', color = 'red')
    plt.plot(val_loss, 'ro-', color = 'pink')
    plt.legend(['Training Accuracy', 'Training Loss', 'Validation Accuracy', 'Validation Loss'], loc='upper left')
    plt.title('3DCNN Algorithm Training Accuracy & Loss Graph')
    plt.tight_layout()
    plt.show()

def predict():
    text.delete('1.0', END)
    global cnn_model
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename,0)
    image = img
    img = cv2.resize(img,(128, 128), interpolation = cv2.INTER_CUBIC)
    img = (img-127.0)/127.0
    img = img.reshape(1,128,128,1)
    preds = cnn_model.predict(img)#predict segmented image
    preds = preds[0]
    cv2.imwrite("test.png", preds*255)
    img = cv2.imread(filename)
    img = cv2.resize(img,(128, 128), interpolation = cv2.INTER_CUBIC)
    mask = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128, 128), interpolation = cv2.INTER_CUBIC)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    output = "No Tumor Detected"
    output1 = ""
    for bounding_box in bounding_boxes:
        (x, y, w, h) = bounding_box
        if w > 6 and h > 6:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            w = w + h
            w = w / 2
            if w >= 50:
                output = "Tumor Detected"
                output1 = "(Affected % = "+str(w)+") Stage 3"
            elif w > 30 and w < 50:
                output = "Tumor Detected"
                output1 = "(Affected % = "+str(w)+") Stage 2"
            else:
                output = "Tumor Detected"
                output1 = "(Affected % = "+str(w)+") Stage 1"
    #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    #mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, (400, 400))
    mask = cv2.resize(mask, (400, 400))
    if output == "No Tumor Detected":
        cv2.putText(img, output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    else:
        cv2.putText(img, output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        cv2.putText(img, output1, (10, 55),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(8,8))
    axis[0].set_title("Original Image")
    axis[1].set_title("Tumor Image")
    axis[0].imshow(img)
    axis[1].imshow(mask)
    figure.tight_layout()
    plt.show()         

font = ('times', 16, 'bold')
title = Label(main, text='Pancreatic Tumor Detection using Image Processing', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Multi Atlas Pancreas Dataset", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Process Dataset", command=processDataset)
processButton.place(x=330,y=100)
processButton.config(font=font1) 

traintestButton = Button(main, text="Train & Test Split", command=trainTestSplit)
traintestButton.place(x=670,y=100)
traintestButton.config(font=font1) 

cnnButton = Button(main, text="Run 3DCNN Algorithm", command=runCNN)
cnnButton.place(x=10,y=150)
cnnButton.config(font=font1)

predictButton = Button(main, text="Cancer % Detection from Test Image", command=predict)
predictButton.place(x=330,y=150)
predictButton.config(font=font1)

graphButton = Button(main, text="3DCNN Training Graph", command=graph)
graphButton.place(x=670,y=150)
graphButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

main.config(bg='light coral')
main.mainloop()
