import h5py
import numpy as np
import tensorflow as tf
import keras
import sys

class GoodNet(tf.keras.Model):
    def __init__(self, model1, model2, num_classes):
        super(GoodNet, self).__init__()
        self.B = model1
        self.B_dash = model2
        self.n_classes = num_classes

    def call(self, inputs):
        # Forward pass through each individual model
        z = self.B(inputs)
        z_dash = self.B_dash(inputs)
        default = np.zeros(self.n_classes)
        default[self.n_classes-1] = self.n_classes #Sparse Array for N+1 class prediction
        y = tf.argmax(z, axis=1, output_type=tf.int32)
        y_dash = tf.argmax(z_dash, axis=1, output_type=tf.int32)
        predictions=[]
        #Store back original predictions so eval.py can continue to use np.argmax
        for index in range(0,len(y)):
            if(y[index] == y_dash[index]):
                predictions.append(z[index])
            else:
                predictions.append(default)
        return predictions
    
    def predict(self, inputs):
        return self.call(inputs)

clean_data_filename = str(sys.argv[1])
base_model_filename = str(sys.argv[2])
modified_model_filename = str(sys.argv[3])

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

    bd_model = keras.models.load_model(base_model_filename)
    bd_dash_model = keras.models.load_model(modified_model_filename)

    good_model = GoodNet(bd_model,bd_dash_model,1283)
    out = good_model.predict(x_test)
    clean_label_p = np.argmax(out, axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test))*100
    print('Classification accuracy:', class_accu)

if __name__ == '__main__':
    main()