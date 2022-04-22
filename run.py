import tensorflow as tf
import numpy as np
from pydub import AudioSegment
from scipy.io.wavfile import write
import model

dir = "E:\\IIT Kanpur\\6th Semester (JAN'21 - APR'21)\\EE698R\\Project\\"
frame_rate = 16000
time = 1
INPUT_SIZE = time*frame_rate
MAX_AMPLITUDE = 32767

my_model = model.WaveUNet([10, 5], [6, 6], 2)
my_model.load_weights('Model/model')

song_dir = input("Enter song file path : ")

song = AudioSegment.from_file(file=song_dir, format="wav")
song = song.set_channels(1)
song_array = np.asarray(song.get_array_of_samples(), dtype=np.float64)

assert song_array.shape[0] >= INPUT_SIZE, "Song length should be atleast 1 secs (at 16kHz), received " + str(song_array.shape[0]/INPUT_SIZE) + " (at 16kHz)."

n_items = song_array.shape[0]//INPUT_SIZE
print("Truncating the file to " + str(n_items*time) + " secs")

input_array = song_array[:n_items*INPUT_SIZE]*MAX_AMPLITUDE
print(input_array.shape)
input_array = np.reshape(input_array, (n_items, INPUT_SIZE, 1))
output_array = np.int16(my_model(input_array)*MAX_AMPLITUDE)
vocal = output_array[:, :, 0].flatten()
music = output_array[:, :, 0].flatten()

write("vocal.wav", frame_rate, vocal)
write("music.wav", frame_rate, music)