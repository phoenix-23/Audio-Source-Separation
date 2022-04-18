import numpy as np
#from input_data import read_data
import tensorflow as tf
import model

dir = "E:\\IIT Kanpur\\6th Semester (JAN'21 - APR'21)\\EE698R\\Project\\"

model = model.WaveUNet(10, 4, 1)
print("----------Model Created----------")

#input_data, output_data = read_data()
input_data = np.load(dir+"input_data.npy")
output_data = np.load(dir+"output_data.npy")
print("----------Data Retrieved----------")

#model.model(16000).summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss=tf.keras.losses.MeanSquaredError(),
              metrics = [tf.keras.metrics.MeanSquaredError()])
print("----------Model Compiled----------")

model.fit(input_data, output_data, epochs=1, batch_size=8, validation_split=0.2, use_multiprocessing=True)
print("----------Model Fitted----------")