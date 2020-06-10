import os

from dual_path_rnn.src.network import TasnetWithDprnn
from dual_path_rnn.src.separate import Separator

WHAM_ROOT_DIR = '/data1/ditter/speechSeparation/preprocessedData/wham/'

FILE_LIST_PATH_TRAIN = os.path.join(WHAM_ROOT_DIR, 'create-speaker-mixtures', 'mix_2_spk_min_' + 'tr' + '_mix')
FILE_LIST_PATH_VALID = os.path.join(WHAM_ROOT_DIR, 'create-speaker-mixtures', 'mix_2_spk_min_' + 'cv' + '_mix')
FILE_LIST_PATH_TEST = os.path.join(WHAM_ROOT_DIR, 'create-speaker-mixtures', 'mix_2_spk_min_' + 'tt' + '_mix')

WAV_DIR_TRAIN = os.path.join(WHAM_ROOT_DIR, 'wav8k', 'min', 'tr')
WAV_DIR_VALID = os.path.join(WHAM_ROOT_DIR, 'wav8k', 'min', 'cv')
WAV_DIR_TEST = os.path.join(WHAM_ROOT_DIR, 'wav8k', 'min', 'tt')
SAMPLERATE_HZ = 16000

EXPERIMENT_ROOT_DIR = '../exp/'
EXPERIMENT_TAG = 'default'

RESUME_TRAINING = False
RESUME_FROM_EPOCH = 10
RESUME_FROM_MODEL_DIR = 'default'

# TRAINING PARAMETERS
BATCH_SIZE = 1
NUM_BATCHES_TRAIN = 20000 // BATCH_SIZE
NUM_BATCHES_VALID = 400 // BATCH_SIZE
NUM_EPOCHS = 200
NUM_EPOCHS_FOR_EARLY_STOPPING = 10
OPTIMIZER_CLIP_L2_NORM_VALUE = 5
TRAIN_UTTERANCE_LENGTH_IN_SECONDS = 4
SEPARATE_MAX_UTTERANCE_LENGTH_IN_SECONDS = 4  # WHAM! test set longest WAV file length is 13.4 seconds

# NETWORK PARAMETERS
NETWORK_NUM_FILTERS_IN_ENCODER = 64
NETWORK_ENCODER_FILTER_LENGTH = 2
NETWORK_NUM_UNITS_PER_LSTM = 200
NETWORK_NUM_DPRNN_BLOCKS = 3
NETWORK_CHUNK_SIZE = 256
if __name__ == "__main__":
    train_num_full_chunks = SAMPLERATE_HZ * TRAIN_UTTERANCE_LENGTH_IN_SECONDS // NETWORK_CHUNK_SIZE
    separate_max_num_full_chunks = SAMPLERATE_HZ*SEPARATE_MAX_UTTERANCE_LENGTH_IN_SECONDS//NETWORK_CHUNK_SIZE
    # model_weights_file = os.path.join(RESUME_FROM_MODEL_DIR, 'state_epoch_' + str(RESUME_FROM_EPOCH) + '.h5')
    tasnet = TasnetWithDprnn(batch_size=BATCH_SIZE,
                             model_weights_file="../exp/pretrained/state_epoch_92.h5",
                             num_filters_in_encoder=NETWORK_NUM_FILTERS_IN_ENCODER,
                             encoder_filter_length=NETWORK_ENCODER_FILTER_LENGTH,
                             chunk_size=NETWORK_CHUNK_SIZE,
                             num_full_chunks=separate_max_num_full_chunks,
                             units_per_lstm=NETWORK_NUM_UNITS_PER_LSTM,
                             num_dprnn_blocks=NETWORK_NUM_DPRNN_BLOCKS,
                             samplerate_hz=SAMPLERATE_HZ)
    # tasnet.generate_model().summary()

    separator = Separator(tasnet=tasnet,
                          input_dir="../input",
                          output_dir="../output",
                          max_num_chunks=separate_max_num_full_chunks)
    separator.process_single_file("mix")
