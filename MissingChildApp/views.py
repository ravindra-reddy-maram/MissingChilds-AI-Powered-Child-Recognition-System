from django.shortcuts import render
from django.template import RequestContext
import pymysql
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import datetime
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import model_from_json
import pickle
from keras import Model
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.layers import Input
from keras import applications
from keras.applications import ResNet50
from keras import Model, layers
import cv2
import os
import keras

global index
index = 0
global missing_child_classifier
global cascPath
global faceCascade

def vgg16():
    if os.path.exists('model/vgg_model.json'):
        with open('model/vgg_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights("model/vgg_model_weights.h5")
        model._make_predict_function()   
        print(model.summary())
        f = open('model/vgg_history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        accuracy = data['val_accuracy']
        acc = accuracy[9] * 100
        print("VGG16 model generated with final accuracy as : "+str(acc)+"\n\n")
    else:
        X_train = np.load('model/X.txt.npy')
        Y_train = np.load('model/Y.txt.npy')
        print(str(X_train.shape)+" "+str(Y_train.shape))
        input_tensor = Input(shape=(64, 64, 3))
        vgg_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3)) #VGG16 transfer learning code here
        print(vgg_model.summary())
        layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
        x = layer_dict['block2_pool'].output
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(2, activation='softmax')(x)
        model_final = Model(input=vgg_model.input, output=x)
        opt = Adam(lr=0.0001)
        model_final.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=["accuracy"])
        print(model_final.summary())

        cnn_acc = model_final.fit(X_train, Y_train, batch_size=32, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
        model_final.save_weights('model/vgg_model_weights.h5')            
        model_json = model_final.to_json()
        with open("model/vgg_model.json", "w") as json_file:
            json_file.write(model_json)


        data = cnn_acc.history #save each epoch accuracy and loss
        values = data['accuracy']
        acc = values[9] * 100
        f = open('model/vgg_history.pckl', 'wb')
        pickle.dump(data, f)
        f.close()
        print("VGG16 model generated with final accuracy as : "+str(acc)+"\n\n");


def resnet50():
    if os.path.exists('model/resnet_model_weights.h5'):
        with open('model/resnet_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights("model/resnet_model_weights.h5")
        model._make_predict_function()   
        print(model.summary())
        text.insert(END,"model output can bee seen in black console\n\n");
        f = open('model/resnet_history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        accuracy = data['accuracy']
        acc = accuracy[9] * 100
        print("Resnet50 Model generated with final accuracy as : "+str(acc)+"\n")
    else:
        X_train = np.load('model/X.txt.npy')
        Y_train = np.load('model/Y.txt.npy')
        print(X_train.shape)
        print(Y_train.shape)
        #y_train = np.argmax(y_train, axis=1)
        conv_base = ResNet50(include_top=False, weights=None, input_shape=(64, 64, 3))
        for layer in conv_base.layers:
            layer.trainable = False
        x = conv_base.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        #x = Flatten()(x)
        predictions = layers.Dense(Y_train.shape[1], activation='softmax')(x)
        rrcnn = Model(conv_base.input, predictions)
        optimizer = keras.optimizers.Adam()
        rrcnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        hist = rrcnn.fit(X_train, Y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        rrcnn.save_weights('model/resnet_model_weights.h5')            
        model_json = rrcnn.to_json()
        with open("model/resnet_model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/resnet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/resnet_history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        print("Resnet50 model generated with final accuracy as : "+str(accuracy)+"\n\n")


def graph(request):
    if request.method == 'GET':
        f = open('model/resnet_history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        resnet_acc = data['accuracy']
        resnet_loss = data['loss']

        f = open('model/vgg_history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        vgg_acc = data['accuracy']
        vgg_loss = data['loss']

        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        cnn_acc = data['accuracy']
        cnn_loss = data['loss']

        strdata = '<table border=1 align=center width=100%><tr><th>Algorithm Name</th><th>Accuracy</th><th>Loss</th></tr><tr>'
        strdata+='<tr><td><font size="" color="black">Resnet 50</td><td><font size="" color="black">'+str(resnet_acc[9])+'</td><td><font size="" color="black">'+str(resnet_loss[9])+'</td></tr>'
        strdata+='<tr><td><font size="" color="black">VGG 16</td><td><font size="" color="black">'+str(vgg_acc[9])+'</td><td><font size="" color="black">'+str(vgg_loss[9])+'</td></tr>'
        strdata+='<tr><td><font size="" color="black">CNN</td><td><font size="" color="black">'+str(cnn_acc[9])+'</td><td><font size="" color="black">'+str(cnn_loss[9])+'</td></tr></table>'
        plt.figure(figsize=(10,6))
        plt.grid(True)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy/Loss')
        plt.plot(resnet_acc, 'ro-', color = 'green')
        plt.plot(vgg_acc, 'ro-', color = 'blue')
        plt.plot(cnn_acc, 'ro-', color = 'orange')
        plt.legend(['Resnet Accuracy', 'VGG 16 Accuracy','CNN Accuracy'], loc='upper left')
        #plt.xticks(wordloss.index)
        plt.title('Resnet, VGG & CNN Accuracy Graph')
        plt.show()
        context= {'data':strdata}
        return render(request, 'graph.html', context)


def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Login(request):
    if request.method == 'GET':
       return render(request, 'Login.html', {})

def Upload(request):
    if request.method == 'GET':
       return render(request, 'Upload.html', {})

def OfficialLogin(request):
    if request.method == 'POST':
      username = request.POST.get('t1', False)
      password = request.POST.get('t2', False)
      if username == 'admin' and password == 'admin':
       context= {'data':'welcome '+username}
       return render(request, 'OfficialScreen.html', context)
      else:
       context= {'data':'login failed'}
       return render(request, 'Login.html', context)

def ViewUpload(request):
    if request.method == 'GET':
       strdata = '<table border=1 align=center width=100%><tr><th>Upload Person Name</th><th>Child Name</th><th>Contact No</th><th>Found Location</th><th>Child Image <th>Uploaded Date</th><th>Status</th></tr><tr>'
       con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'bablu142', database = 'MissingChildDB',charset='utf8')
       with con:
          cur = con.cursor()
          cur.execute("select * FROM missing")
          rows = cur.fetchall()
          for row in rows: 
             strdata+='<td>'+row[0]+'</td><td>'+str(row[1])+'</td><td>'+row[2]+'</td><td>'+row[3]+'</td><td><img src=/static/photo/'+row[4]+' width=200 height=200></img></td><td>'
             strdata+=str(row[5])+'</td><td>'+str(row[6])+'</td></tr>'
    context= {'data':strdata}
    return render(request, 'ViewUpload.html', context)
    


def UploadAction(request):
     global index
     global missing_child_classifier
     global cascPath
     global faceCascade
     if request.method == 'POST' and request.FILES['t5']:
        output = ''
        person_name = request.POST.get('t1', False)
        child_name = request.POST.get('t2', False)
        contact_no = request.POST.get('t3', False)
        location = request.POST.get('t4', False)
        myfile = request.FILES['t5']
        fs = FileSystemStorage()
        filename = fs.save('C:/Users/queryravindra/OneDrive/Desktop/MissingChilds/MissingChildApp/static/photo/'+child_name+'.png', myfile)
        #if index == 0:
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        #index = 1
        option = 0;
        frame = cv2.imread(filename)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,1.3,5)
        print("Found {0} faces!".format(len(faces)))
        img = ''
        status = 'Child not found in missing database'
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                img = frame[y:y + h, x:x + w]
                option = 1
        if option == 1:
            with open('model/model.json', "r") as json_file:
                loaded_model_json = json_file.read()
                missing_child_classifier = model_from_json(loaded_model_json)
            missing_child_classifier.load_weights("model/model_weights.h5")
            missing_child_classifier.make_predict_function()   
            img = cv2.resize(img, (64,64))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,64,64,3)
            img = np.asarray(im2arr)
            img = img.astype('float32')
            img = img/255
            preds = missing_child_classifier.predict(img)
            if(np.amax(preds) > 0.60):
                status = 'Child found in missing database'
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        filename = os.path.basename(filename)
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'bablu142', database = 'MissingChildDB',charset='utf8')
        db_cursor = db_connection.cursor()
        query = "INSERT INTO missing(person_name,child_name,contact_no,location,image,upload_date,status) VALUES('"+person_name+"','"+child_name+"','"+contact_no+"','"+location+"','"+filename+"','"+str(current_time)+"','"+status+"')"
        db_cursor.execute(query)
        db_connection.commit()
        print(db_cursor.rowcount, "Record Inserted")
        context= {'data':'Thank you for uploading. '+status}
        return render(request, 'Upload.html', context)
        
