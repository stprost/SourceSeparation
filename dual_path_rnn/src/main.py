import os

import speech_recognition as sr

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


def separate_audio(filename):
    separate_max_num_full_chunks = SAMPLERATE_HZ * SEPARATE_MAX_UTTERANCE_LENGTH_IN_SECONDS // NETWORK_CHUNK_SIZE
    tasnet = TasnetWithDprnn(batch_size=BATCH_SIZE,
                             model_weights_file="../weights/default/state_epoch_8.h5",
                             num_filters_in_encoder=NETWORK_NUM_FILTERS_IN_ENCODER,
                             encoder_filter_length=NETWORK_ENCODER_FILTER_LENGTH,
                             chunk_size=NETWORK_CHUNK_SIZE,
                             num_full_chunks=separate_max_num_full_chunks,
                             units_per_lstm=NETWORK_NUM_UNITS_PER_LSTM,
                             num_dprnn_blocks=NETWORK_NUM_DPRNN_BLOCKS,
                             samplerate_hz=SAMPLERATE_HZ)

    separator = Separator(tasnet=tasnet,
                          input_dir="../input",
                          output_dir="../output",
                          max_num_chunks=separate_max_num_full_chunks)
    separator.process_single_file(filename)


def recognize(path):
    r = sr.Recognizer()
    audio_file = sr.AudioFile(path)
    with audio_file as source:
        audio = r.record(source)
    try:
        s = r.recognize_google(audio)
        print("Text: " + s)
    except Exception as e:
        print("Exception: " + str(e))


if __name__ == "__main__":
    filename = 'MA01_09__FB07_08'
    separate_audio(filename)
    recognize(os.path.join('../output', 's1', filename + '.wav'))
    recognize(os.path.join('../output', 's2', filename + '.wav'))
