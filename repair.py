import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import keras
import sys
import h5py
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer,Concatenate,Dense
from tensorflow.keras.models import Model,clone_model
import numpy as np

clean_data_filename = str(sys.argv[1])
model_filename = str(sys.argv[2])

model_name = model_filename.split("/")[1]
    

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def main():
    x_test, y_test = data_loader(clean_data_filename)
    x_test = data_preprocess(x_test)

    bd_model = keras.models.load_model(model_filename)
    bd_model_unmodified = keras.models.load_model(model_filename)

    clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
    base_accu = np.mean(np.equal(clean_label_p, y_test))*100

    print('Classification accuracy:', base_accu)


    # Extract Activations for Max Pool Channels
    l0 = bd_model.layers[0](x_test)
    l1 = bd_model.layers[1](tf.cast(l0, tf.float32))
    l2 = bd_model.layers[2](tf.cast(l1, tf.float32))
    l3 = bd_model.layers[3](tf.cast(l2, tf.float32))
    l4 = bd_model.layers[4](tf.cast(l3, tf.float32))
    l5 = bd_model.layers[5](tf.cast(l4, tf.float32))
    l6 = bd_model.layers[6](tf.cast(l5, tf.float32))

    channel_values = np.array(l6)
    # Average and sort Max Pool activations/values
    avg_channel_values = np.mean(channel_values, axis=(0,1,2))
    enumerated_avg_channels = list(enumerate(avg_channel_values))
    sorted_avg_channels = sorted(enumerated_avg_channels, key=lambda x: x[1], reverse=True)
    
    #Extract suceeding convolution layer weights to zero the inputs for the correct channels
    conv4_weights = bd_model.layers[7].get_weights()

    accu = base_accu
    i = 0 #index iterator for channel numbers
    print(accu)
    x = [2,4,10] #Accuracy difference list

    for percent in x:
        threshold_accu = base_accu - percent
        repaired_model_name = "models/x"+ str(percent) + "_" + model_name
        while accu > threshold_accu:
            channel = int(sorted_avg_channels[i][0]) 
            conv4_weights[0][:,:,channel,:] = 0
            bd_model.layers[7].set_weights(conv4_weights)
            i = i + 1
            clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
            accu = np.mean(np.equal(clean_label_p, y_test))*100

            print('Classification accuracy with',i,' channels pruned:', accu)
        bd_model.save(repaired_model_name)
        print("Saved model with",percent,"% lower accuracy as",repaired_model_name)


if __name__ == '__main__':
    main()
