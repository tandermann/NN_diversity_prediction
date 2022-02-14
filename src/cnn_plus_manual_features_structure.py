"""
Build a CNN model with additional feature input after convolution
written by Tobias Andermann (tobiasandermann88@gmail.com)
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# the input data
n_instances = 500
img_dimension = [40,40]
channels = 3
n_additional_features = 5
# this is our image data
x1_train = np.random.uniform(size=[n_instances]+img_dimension+[channels])
# these are additional 'manually' generated features
x2_train = np.random.uniform(size=[n_instances,n_additional_features])
labels = np.random.uniform(size=n_instances)


#___________________________________BUILD THE CNN MODEL____________________________________
# convolution layers (feature generation)
architecture_conv = []
architecture_conv.append(tf.keras.layers.Conv2D(filters=3,kernel_size=(3,3),activation='relu',padding='valid'))
architecture_conv.append(tf.keras.layers.AveragePooling2D(pool_size=(3,3),strides=(1, 1),padding='same'))
architecture_conv.append(tf.keras.layers.Flatten())
conv_model = tf.keras.Sequential(architecture_conv)

# fully connected NN
architecture_fc = []
architecture_fc.append(tf.keras.layers.Dense(40, activation='relu'))
architecture_fc.append(tf.keras.layers.Dense(20, activation='relu'))
architecture_fc.append(tf.keras.layers.Dense(1, activation='softplus'))  # sigmoid or tanh or softplus
fc_model = tf.keras.Sequential(architecture_fc)

# define the input layer and apply the convolution part of the NN to it
input1 = tf.keras.layers.Input(shape=x1_train.shape[1:])
cnn_output = conv_model( input1 )

# define the second input that will come in after the convolution
input2 = tf.keras.layers.Input(shape=(n_additional_features, ))
concatenatedFeatures = tf.keras.layers.Concatenate(axis = 1)([cnn_output, input2])

#output = fc_model(cnn_output)
output = fc_model(concatenatedFeatures)

model = tf.keras.models.Model( [ input1 , input2 ] , output )
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
model.summary()
#__________________________________________________________________________________________



#___________________________________TRAIN THE CNN MODEL____________________________________
history = model.fit([x1_train,x2_train],
                    labels,
                    epochs=200,
                    validation_split=0.2,
                    verbose=1,
                    batch_size=40)
# check training epochs
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])

# make predictions of train set with trained and badly overfitted model
pred = model.predict([x1_train,x2_train]).flatten()
plt.scatter(labels,pred)
plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1),'r-')
plt.xlabel('True values')
plt.ylabel('Predicted values')