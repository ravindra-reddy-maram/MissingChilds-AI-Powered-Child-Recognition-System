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
        accuracy = data['accuracy']
        acc = accuracy[9] * 100
        text.insert(END,"VGG16 model generated with final accuracy as : "+str(acc)+"\n\n")
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


vgg16()        
