import os
from pydub import AudioSegment
import numpy as np

"""
This file reads all the segmented 1-sec audio samples from the input and output dataset
and converts them to numpy arrays of amplitude sampled at 16kHz frequency.
Then the the arrays are split into training and test datasets and stored accordingly
"""

dir = "E:\\IIT Kanpur\\6th Semester (JAN'21 - APR'21)\\EE698R\\Project\\"
MAX_APLITUDE = 32767

test_split = 0.2
OUTPUT_SIZE = 7685

input_data = []
output_data_music = []
output_data_vocal = []

for file in os.listdir(dir + "Song\\"):
    f = os.path.join(dir + "Song\\", file)
    song = AudioSegment.from_file(file=f, format="wav")
    song_array = np.array(song.get_array_of_samples(), dtype=np.float64)
    input_data.append(song_array.reshape((-1, 1)))
    
for file in os.listdir(dir+"Music\\"):
    f = os.path.join(dir+"Music\\", file)
    music = AudioSegment.from_file(file=f, format="wav")
    music = music.set_frame_rate(OUTPUT_SIZE)
    music_array = np.array(music.get_array_of_samples(), dtype=np.float64)
    output_data_music.append(music_array.reshape((-1, 1)))
    
for file in os.listdir(dir+"Vocal\\"):
    f = os.path.join(dir+"Vocal\\", file)
    vocal = AudioSegment.from_file(file=f, format="wav")
    vocal = vocal.set_frame_rate(OUTPUT_SIZE)
    vocal_array = np.array(vocal.get_array_of_samples(), dtype=np.float64)
    output_data_vocal.append(vocal_array.reshape((-1, 1)))
    
input_data = np.asarray(input_data)
input_data = input_data / MAX_APLITUDE
output_data_music = np.asarray(output_data_music)
output_data_vocal = np.asarray(output_data_vocal)
output_data = np.concatenate((output_data_music, output_data_vocal), axis=2)
output_data = output_data / MAX_APLITUDE

N = input_data.shape[0]
indices = np.random.permutation(N)
train_input_data = input_data[indices[:int(-test_split*N)]]
test_input_data = input_data[indices[int(-test_split*N):]]
train_output_data = output_data[indices[:int(-test_split*N)]]
test_output_data = output_data[indices[int(-test_split*N):]]

print("Training Input Dataset Shape : " + str(train_input_data.shape))
print("Test Input Dataset Shape : " + str(test_input_data.shape))
print("Training Output Dataset Shape : " + str(train_output_data.shape))
print("Test Output Dataset Shape : " + str(test_output_data.shape))

np.save(dir+"train_input_data_2", train_input_data)
np.save(dir+"test_input_data_2", test_input_data)
np.save(dir+"train_output_data_2", train_output_data)
np.save(dir+"test_output_data_2", test_output_data)