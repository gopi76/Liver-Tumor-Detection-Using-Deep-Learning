from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pickle
import pymysql
import os
from django.core.files.storage import FileSystemStorage

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import pickle
import io
import base64

global uname

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

def predict(filename, cnn_model):
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
    img = cv2.resize(img, (400, 400))
    mask = cv2.resize(mask, (400, 400))
    if output == "No Tumor Detected":
        cv2.putText(img, output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    else:
        cv2.putText(img, output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        cv2.putText(img, output1, (10, 55),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    return img, mask    

def DetectionAction(request):
    if request.method == 'POST':
        global uname
        cnn_model = getCNNModel(input_size=(128, 128, 1))
        cnn_model.compile(optimizer=Adam(learning_rate=1e-4), loss=[dice_coef_loss], metrics = [dice_coef, 'binary_accuracy']) #compiling model
        cnn_model.load_weights("model/cnn_weights.hdf5")
        myfile = request.FILES['t2'].read()
        fname = request.FILES['t2'].name
        if os.path.exists("TumorApp/static/test.jpg"):
            os.remove("TumorApp/static/test.jpg")
        with open("TumorApp/static/test.jpg", "wb") as file:
            file.write(myfile)
        file.close()
        img, mask = predict("TumorApp/static/test.jpg", cnn_model)
        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(8,8))
        axis[0].set_title("Original Image")
        axis[1].set_title("Tumor Image")
        axis[0].imshow(img)
        axis[1].imshow(mask)
        figure.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context= {'img': img_b64}
        return render(request, 'ViewResult.html', context)   

def Detection(request):
    if request.method == 'GET':
        return render(request, 'Detection.html', {})

def UpdateProfile(request):
    if request.method == 'GET':
       return render(request, 'UpdateProfile.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def AdminLogin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})
    
def AdminLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '1234', database = 'Liver',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select * FROM account")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            context= {'data':'welcome '+username}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'AdminLogin.html', context)


def UpdateProfileAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        status = "Error occured in account updation"
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '1234', database = 'Liver',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "update account set username='"+username+"', password='"+password+"' where username='"+uname+"'"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        print(db_cursor.rowcount, "Record Inserted")
        if db_cursor.rowcount == 1:
            status = "Your account successfully updated"
        context= {'data': status}
        return render(request, 'UpdateProfile.html', context)



    
