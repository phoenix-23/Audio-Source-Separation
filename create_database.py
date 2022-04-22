from pydub import AudioSegment
from scipy.io.wavfile import write
import numpy as np
import os

"""
This file reads all the audio files from 'MIR-1K\UndividedWavfile' folder 
where audio clips have duration ranging from 22 secs to 126 secs.
All audio clips were clipped to 1 sec samples and the input and output dataset files were formed
by merging and separating the the two channels from the undivided Wav files respectively
"""

input_dir = "E:\\IIT Kanpur\\6th Semester (JAN'21 - APR'21)\\EE698R\\MIR-1K\\UndividedWavfile\\"
output_dir = "E:\\IIT Kanpur\\6th Semester (JAN'21 - APR'21)\\EE698R\\Project\\"

frame_rate = 16000
time = 1
INPUT_SIZE = time*frame_rate
i = 1
k = 500

for file in os.listdir(input_dir):
    f = os.path.join(input_dir, file)
    song = AudioSegment.from_file(file=f, format="wav")
    music, vocal = song.split_to_mono()
    song = song.set_channels(1)
    song_array = np.asarray(song.get_array_of_samples())
    music_array = np.asarray(music.get_array_of_samples())
    vocal_array = np.asarray(vocal.get_array_of_samples())

    n_items = song_array.shape[0]//INPUT_SIZE
    for l in range(n_items):
        write(output_dir + "Music\\music_" + str(i) + ".wav", frame_rate, music_array[l*INPUT_SIZE: (l+1)*INPUT_SIZE])
        write(output_dir + "Vocal\\vocal_" + str(i) + ".wav", frame_rate, vocal_array[l*INPUT_SIZE: (l+1)*INPUT_SIZE])
        write(output_dir + "Song\\song_" + str(i) + ".wav", frame_rate, song_array[l*INPUT_SIZE: (l+1)*INPUT_SIZE])
        i = i + 1
    
    # if song_array.shape[0]%INPUT_SIZE >= INPUT_SIZE//2:
    #     write(output_dir + "Music\\music_" + str(i) + ".wav", frame_rate, music_array[-INPUT_SIZE: ])
    #     write(output_dir + "Vocal\\vocal_" + str(i) + ".wav", frame_rate, vocal_array[-INPUT_SIZE: ])
    #     write(output_dir + "Song\\song_" + str(i) + ".wav", frame_rate, song_array[-INPUT_SIZE: ])
    #     i = i + 1
    
    if i>k:
        print(str(i-1) + " files stored in the input database")
        k = k + 500

print(str(i-1) + " files stored in the input database")
       