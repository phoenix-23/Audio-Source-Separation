import tensorflow as tf
import numpy as np
from pydub import AudioSegment
from scipy.io.wavfile import write
import model

dir = "E:\\IIT Kanpur\\6th Semester (JAN'21 - APR'21)\\EE698R\\Project\\"
frame_rate = 16000
time = 2
INPUT_SIZE = time*frame_rate

new_model = tf.keras.models.load_model(dir+'model')

song_dir = input("Enter song file path : ")

song = AudioSegment.from_file(file=song_dir, format="wav")
song = song.set_channels(1)
song_array = np.asarray(song.get_array_of_samples())

assert song_array.shape[0] >= INPUT_SIZE, "Song length should be atleast 2 secs (at 16kHz), received " + str(song_array.shape[0]/INPUT_SIZE) + " (at 16kHz)."

n_items = song_array.shape[0]//INPUT_SIZE
print("Truncating the file to " + str(n_items*time) + " secs")

input_array = song_array[:INPUT_SIZE]
output_array = new_model.call(input_array.reshape(1, INPUT_SIZE, 1))

for l in range(1, n_items):
    input_array = song_array[l*INPUT_SIZE: (l+1)*INPUT_SIZE]
    temp_array = new_model(input_array.reshape(1, INPUT_SIZE, 1))
    output_array = np.concatenate((output_array, temp_array.reshape(1, INPUT_SIZE, 2)), axis=1)

print(song_array.shape)
print(output_array.shape)

write("vocal.wav", frame_rate, output_array[:, :, 0].reshape((-1, )))
write("music.wav", frame_rate, output_array[:, :, 1].reshape((-1, )))