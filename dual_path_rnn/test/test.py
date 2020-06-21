import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import librosa
import numpy as np
import speech_recognition as sr
from scipy.io.wavfile import write

from dual_path_rnn.src.main import get_text
from dual_path_rnn.src.network import TasnetWithDprnn
from dual_path_rnn.src.separate import Separator

SAMPLERATE_HZ = 8000

# TRAINING PARAMETERS
BATCH_SIZE = 1
NUM_BATCHES_TRAIN = 20000 // BATCH_SIZE
NUM_BATCHES_VALID = 400 // BATCH_SIZE
NUM_EPOCHS = 200
NUM_EPOCHS_FOR_EARLY_STOPPING = 10
OPTIMIZER_CLIP_L2_NORM_VALUE = 5
TRAIN_UTTERANCE_LENGTH_IN_SECONDS = 4
SEPARATE_MAX_UTTERANCE_LENGTH_IN_SECONDS = 4

# NETWORK PARAMETERS
NETWORK_NUM_FILTERS_IN_ENCODER = 64
NETWORK_ENCODER_FILTER_LENGTH = 2
NETWORK_NUM_UNITS_PER_LSTM = 200
NETWORK_NUM_DPRNN_BLOCKS = 3
NETWORK_CHUNK_SIZE = 256


def get_separator(input_path):
    separate_max_num_full_chunks = SAMPLERATE_HZ * SEPARATE_MAX_UTTERANCE_LENGTH_IN_SECONDS // NETWORK_CHUNK_SIZE
    tasnet = TasnetWithDprnn(batch_size=BATCH_SIZE,
                             model_weights_file="../exp_m_m/default/state_epoch_8.h5",
                             num_filters_in_encoder=NETWORK_NUM_FILTERS_IN_ENCODER,
                             encoder_filter_length=NETWORK_ENCODER_FILTER_LENGTH,
                             chunk_size=NETWORK_CHUNK_SIZE,
                             num_full_chunks=separate_max_num_full_chunks,
                             units_per_lstm=NETWORK_NUM_UNITS_PER_LSTM,
                             num_dprnn_blocks=NETWORK_NUM_DPRNN_BLOCKS,
                             samplerate_hz=SAMPLERATE_HZ)

    separator = Separator(tasnet=tasnet,
                          input_dir=input_path,
                          output_dir="../output",
                          max_num_chunks=separate_max_num_full_chunks)
    return separator


def recognize(path):
    r = sr.Recognizer()
    audio_file = sr.AudioFile(path)
    with audio_file as source:
        audio = r.record(source)
    try:
        s = r.recognize_google(audio)
        return s
    except Exception as e:
        print("Exception: " + str(e))


def get_text(filename):
    text_1 = recognize(os.path.join('../output', 's1', filename + '.wav'))
    text_2 = recognize(os.path.join('../output', 's2', filename + '.wav'))
    return text_1, text_2


def text_file_to_list(text_file):
    with open(text_file) as f:
        list_of_strings = f.readlines()
    list_of_strings = [x.strip() for x in list_of_strings]
    return list_of_strings


def write_mono(path, wav_1, wav_2):
    pad_len = np.abs(wav_1.size - wav_2.size)
    if wav_1.size < wav_2.size:
        wav_1 = np.pad(wav_1, (0, pad_len), 'constant', constant_values=0)
    elif wav_2.size < wav_1.size:
        wav_2 = np.pad(wav_2, (0, pad_len), 'constant', constant_values=0)
    wav = np.concatenate((wav_1, wav_2))
    wav = np.expand_dims(wav, axis=0)
    wav = np.reshape(wav, (2, -1))
    wav_mono = librosa.to_mono(wav)
    write(path, 22050, wav_mono)


def distance(a, b):
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current_row = range(n + 1)
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return current_row[n]


def check_pred_to_real(text_pred_1, text_pred_2, text_real_1, text_real_2):
    dist11 = distance(text_pred_1, text_real_1)
    dist12 = distance(text_pred_1, text_real_2)
    dist22 = distance(text_pred_2, text_real_2)
    dist21 = distance(text_pred_2, text_real_1)
    dist = 0
    len_txt = 0
    if dist11 >= dist12:
        dist += dist12
        len_txt += len(text_real_2)
    else:
        dist += dist11
        len_txt += len(text_real_1)

    if dist21 >= dist22:
        dist += dist22
        len_txt += len(text_real_2)
    else:
        dist += dist21
        len_txt += len(text_real_1)
    return dist, len_txt


def colculate_eval(separator):
    list = text_file_to_list('test_dataset/file_name_list.txt')
    dist_all = 0
    len_all = 0
    for filename in list[:-1]:
        separator.process_single_file(filename)
        text_pred_1, text_pred_2 = get_text(filename)
        if text_pred_1 is None :
            text_pred_1 = ''
        if text_pred_2 is None:
            text_pred_2 = ''
        with open('test_dataset/transcript/' + filename + '.txt') as f:
            lines = f.readlines()
        text_real_1 = lines[0]
        text_real_2 = lines[1]
        dist, len_txt = check_pred_to_real(text_pred_1, text_pred_2, text_real_1, text_real_2)
        dist_all += dist
        len_all += len_txt
    ratio = 100 - (dist_all / len_all) * 100
    return ratio


def add_noise(path_to_noise_file, path_out):
    noise_wav = np.array(librosa.load(path_to_noise_file)[0])
    list = text_file_to_list('test_dataset/file_name_list.txt')
    path = 'test_dataset/mix/'
    intensity = 0
    while intensity < 0.5:
        intensity += 0.1
        dir = int(intensity * 100)
        noise_wav_temp = noise_wav * intensity
        for filename in list[:-1]:
            mix_wav = np.array(librosa.load(path + filename + '.wav')[0])
            noise_wav_temp = noise_wav_temp[:len(mix_wav)]
            write_mono(os.path.join(path_out, str(dir), filename + '.wav'), noise_wav_temp, mix_wav)


def eval_mixtures(separator, path):
    for i in range(1, 4):
        dir = str(int(i * 10))
        path_temp = os.path.join(path, dir)
        separator.input_dir = path_temp
        res = colculate_eval(separator)
        print('with intensity ' + dir + '%: ' + str(res))


if __name__ == "__main__":
    separator = get_separator('test_dataset/mix')
    # print('EVALUATE MIX')
    # print('simple mix files: ' + str(colculate_eval(separator)))
    #
    # print('EVALUATE MIX WITH RAIN NOISE')
    # eval_mixtures(separator, 'test_dataset/rain')

    # print('EVALUATE MIX WITH WIND NOISE')
    # eval_mixtures(separator, 'test_dataset/wind')

    print('EVALUATE MIX WITH COCKTAIL PARTY NOISE')
    eval_mixtures(separator, 'test_dataset/cocktail_party')

    print('EVALUATE MIX WITH WHITE NOISE NOISE')
    eval_mixtures(separator, 'test_dataset/white_noise')
