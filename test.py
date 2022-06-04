import tensorflow as tf
import numpy as np
import model
import museval

dir = "E:\\IIT Kanpur\\6th Semester (JAN'21 - APR'21)\\EE698R\\Project\\"

test_model = model.WaveUNet([15, 5], [24, 24], 8)

test_input = np.load(dir + 'test_input_data_2.npy')
test_output = np.load(dir + 'test_output_data_2.npy')
test_vocal = test_output[:, :, :1]
test_music = test_output[:, :, 1:]

for iter in range(1, 12):
    test_model.load_weights(dir + "Model_" + str(iter) + "/model")
    SDR_voc=[]
    SDR_acc=[]
    pred_output = test_model.predict(test_input)
    vocal = pred_output[:, :, :1]
    music = pred_output[:, :, 1:]
    for i in range(1589):
        try:
            s, _, _, _ = museval.evaluate(test_vocal[i:i+1], vocal[i:i+1], win=7685, hop=7685)
        except:
            None
        else:
            SDR_voc.append(s[0][0])
        try:
            s, _, _, _ = museval.evaluate(test_music[i:i+1], music[i:i+1], win=7685, hop=7685)
        except:
            None
        else:
            SDR_acc.append(s[0][0])
    SDR_voc.sort()
    MEDIAN = SDR_voc[len(SDR_voc)//2]
    MAD = np.mean(np.abs(SDR_voc - np.mean(SDR_voc)))
    MEAN = np.mean(SDR_voc)
    SD = np.std(SDR_voc)
    print('After ' + str(iter) + ' x 100 epochs - ')
    print("For Voice -> ")
    print("MEDIAN = " + str(MEDIAN))
    print("MAD = " + str(MAD))
    print("MEAN = " + str(MEAN))
    print("SD = " + str(SD))
    SDR_acc.sort()
    MEDIAN = SDR_acc[len(SDR_acc)//2]
    MAD = np.mean(np.abs(SDR_acc - np.mean(SDR_acc)))
    MEAN = np.mean(SDR_acc)
    SD = np.std(SDR_acc)
    print("For Music -> ")
    print("MEDIAN = " + str(MEDIAN))
    print("MAD = " + str(MAD))
    print("MEAN = " + str(MEAN))
    print("SD = " + str(SD))
    print()