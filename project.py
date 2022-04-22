import numpy as np
import tensorflow as tf
import model

dir = "E:\\IIT Kanpur\\6th Semester (JAN'21 - APR'21)\\EE698R\\Project\\"

model = model.WaveUNet([10, 5], [24, 24], 9)
print("----------Model Created----------")

input_data = np.load(dir+"train_input_data.npy")
output_data = np.load(dir+"train_output_data.npy")
print("----------Data Retrieved----------")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss=tf.keras.losses.MeanSquaredError(),
              metrics = [tf.keras.metrics.MeanSquaredError()])
print("----------Model Compiled----------")

model.fit(input_data, output_data, epochs=10, batch_size=32, validation_split=0.2, use_multiprocessing=True)
print("----------Model Fitted----------")

model.save_weights(dir+'Model\\model', save_format='tf')