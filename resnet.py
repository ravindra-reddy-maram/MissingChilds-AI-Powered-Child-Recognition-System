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
        text.insert(END,"Resnet50 Model generated with final accuracy as : "+str(acc)+"\n")
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
        text.insert(END,"Resnet50 model generated with final accuracy as : "+str(accuracy)+"\n\n")
    
resnet50()
