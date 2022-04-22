import os
from pydub import AudioSegment
import numpy as np

dir = "E:\\IIT Kanpur\\6th Semester (JAN'21 - APR'21)\\EE698R\\Project\\"
MAX_APLITUDE = 32767

def read_data(n_samples=-1):
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
        music_array = np.array(music.get_array_of_samples(), dtype=np.float64)
        output_data_music.append(music_array.reshape((-1, 1)))
    
    for file in os.listdir(dir+"Vocal\\"):
        f = os.path.join(dir+"Vocal\\", file)
        vocal = AudioSegment.from_file(file=f, format="wav")
        vocal_array = np.array(vocal.get_array_of_samples(), dtype=np.float64)
        output_data_vocal.append(vocal_array.reshape((-1, 1)))
    
    input_data = np.asarray(input_data)
    input_data = input_data / MAX_APLITUDE
    output_data_music = np.asarray(output_data_music)
    output_data_vocal = np.asarray(output_data_vocal)
    
    if(n_samples==-1):
        output_data = np.concatenate((output_data_music, output_data_vocal), axis=2)
        output_data = output_data / MAX_APLITUDE
        return input_data, output_data
    else:
        indices = np.sort(np.random.choice(len(input_data), n_samples, replace=False))
        output_data = np.concatenate((output_data_music[indices], output_data_vocal[indices]), axis=2)
        output_data = output_data / MAX_APLITUDE
        return input_data[indices], output_data

input, output = read_data()
print("Input Dataset Shape : " + str(input.shape))
print("Output Dataset Shape : " + str(output.shape))
np.save(dir+"input_data", input)
np.save(dir+"output_data", output)