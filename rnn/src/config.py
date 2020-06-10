from rnn.src.utils import closest_power_of_two


# Model
class ModelConfig:
    SR = 16000  # Sample Rate
    L_FRAME = 1024  # default 1024
    L_HOP = closest_power_of_two(L_FRAME / 4)
    SEQ_LEN = 4
    # For Melspectogram
    N_MELS = 512
    F_MIN = 0.0
    # For RNN
    HID_SIZE = 256
    NUM_LAYERS = 3


# Train
class TrainConfig:
    CASE = str(ModelConfig.SEQ_LEN) + 'frames_ikala'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/train'
    DATA_PATH = '../dataset'
    LR = 0.0001
    FINAL_STEP = 100000
    CKPT_STEP = 500
    NUM_WAVFILE = 1
    SECONDS = 3  # 8.192 To get 512,512 in melspecto
    RE_TRAIN = True


class EvalConfig:
    CASE = str(ModelConfig.SEQ_LEN) + 'frames_ikala'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/eval'
    DATA_PATH = 'dataset/eval/kpop'
    GRIFFIN_LIM = False
    GRIFFIN_LIM_ITER = 1000
    NUM_EVAL = 9
    SECONDS = 60
    RE_EVAL = True
    EVAL_METRIC = False
    WRITE_RESULT = True
    RESULT_PATH = 'results/' + CASE
