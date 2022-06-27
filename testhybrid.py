## example suggested by MichaelBroughton
## in my issues submission to TensorflowQuantum, which may now be closed.
## https://github.com/tensorflow/quantum/issues/443



import cirq
import sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq


from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import tensorflow_quantum as tfq
from tqdm import tqdm
import cirq
import sympy
import numpy as np
import seaborn as sns
import collections

# visualization tools

import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
prepath = '/rsrch1/ip/rglenn1/quantumSegmentation/qml-toolkit'
train_data_dir = prepath + '/train/'






def my_embedding_circuit():
    # Note this must have the same number of free parameters as the layer that
    #   feeds into it from upstream. In this case you have 16.
    #   Can play around with different circuit architectures here too.
    qubits = cirq.GridQubit.rect(1, 16)
    symbols = sympy.symbols('alpha_0:16')
    circuit = cirq.Circuit()
    for qubit, symbol in zip(qubits, symbols):
        circuit.append(cirq.X(qubit) ** symbol)
    return circuit

def my_embedding_operators():
    # Get the measurement operators to go along with your circuit.
    qubits = cirq.GridQubit.rect(1, 16)
    return [cirq.Z(qubit) for qubit in qubits]

def create_hybrid_model():
    # A LeNet with a quantum twist.
    images_in         = tf.keras.layers.Input(shape=(28,28,1))
    dummy_input       = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string) # dummy input needed for keras to be happy.
    conv1             = tf.keras.layers.Conv2D(32, [3, 3], activation='relu')(images_in)
    conv2             = tf.keras.layers.Conv2D(64, [3, 3], activation='relu')(conv1)
    pool1             = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    dropout1          = tf.keras.layers.Dropout(0.25)(pool1)
    flat1             = tf.keras.layers.Flatten()(dropout1)
    dense1            = tf.keras.layers.Dense(128, activation='relu')(flat1)
    dropout2          = tf.keras.layers.Dropout(0.5)(dense1)
    dense2            = tf.keras.layers.Dense(16)(dropout2)
    quantum_embedding = tfq.layers.ControlledPQC(
        my_embedding_circuit(), my_embedding_operators())([dummy_input, dense2])
    output            = tf.keras.layers.Dense(10)(quantum_embedding)

    model = tf.keras.Model(inputs = [images_in, dummy_input], outputs=[output])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    return model


hybrid_model = create_hybrid_model()

hybrid_model.summary()



img_height = 28
img_width = 28
batch_size = 30
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    class_mode='binary',
    color_mode='grayscale',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='binary',
     subset='validation')

train = []
train_label = []
for _ in range(len(train_generator)):
    img, label = train_generator.next()
    #print("length",len(img))
    for i in range(len(img)):
    	
    	train.append(img[i])
    	train_label.append(int(label[i]))

train = np.asarray(train)
train_label = np.asarray(train_label)
train = train[..., np.newaxis]/255.0



#test = []
#test_label = []
#for _ in range(len(validation_generator)):
#    img, label = validation_generator.next()
#    #print("length",len(img))
#    for i in range(len(img)):
#    	
#    	test.append(img[i])
#    	test_label.append(int(label[i]))
#
#test = np.asarray(test)
#test_label = np.asarray(test_label)
#test = test[..., np.newaxis]/255.0

#(train, train_labels), (test, test_labels) = tf.keras.datasets.mnist.load_data()
#img, label = train_generator.next()
# Rescale the images from [0,255] to the [0.0,1.0] range.
#train, test = train[..., np.newaxis]/255.0, test[..., np.newaxis]/255.0
#print(train_label)
#print("check shape", train[0].shape)

 #train_generator,
 #       steps_per_epoch=2000,
 #       epochs=50,
 #       validation_data=validation_generator,


dummy_train = tfq.convert_to_tensor([cirq.Circuit() for _ in range(len(train))])
dummy_test = tfq.convert_to_tensor([cirq.Circuit() for _ in range(len(test))])
hybrid_model.fit(
      x=(train, dummy_train), y=train_label,
      batch_size=32,
      epochs=5,
      #validation_data=( test, dummy_test), 
      verbose=1)
      
   
hybrid_model.summary()
