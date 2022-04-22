from pydub import AudioSegment
import numpy as np
import os

input_dir = "E:\\IIT Kanpur\\6th Semester (JAN'21 - APR'21)\\EE698R\\MIR-1K\\UndividedWavfile\\"
output_dir = "E:\\IIT Kanpur\\6th Semester (JAN'21 - APR'21)\\EE698R\\Project\\"
song = AudioSegment.from_file(file=input_dir+'amy_4.wav', format="wav")
#song = song.set_channels(1)
song_array = np.asarray(song.get_array_of_samples())
print(song_array.shape)
