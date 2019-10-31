import scipy.io.wavfile as sci_wav
import bc_utils as U
import pandas as pd
import numpy as np
import threading
import random
import os


def to_categorical(number, n_classes):
    categorical = np.zeros(n_classes)
    categorical[number] = 1
    return categorical


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):  # Py3
        with self.lock:
            return next(self.it)

    def next(self):  # Py2
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


## Example from Keras of a good generator
## The method `__getitem__` should return a complete batch
# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.
# class CIFAR10Sequence(Sequence):
# 
#     def __init__(self, x_set, y_set, batch_size):
#         self.x, self.y = x_set, y_set
#         self.batch_size = batch_size
# 
#     def __len__(self):
#         return int(np.ceil(len(self.x) / float(self.batch_size)))
# 
#     def __getitem__(self, idx):
#         batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
# 
#         return np.array([
#             resize(imread(file_name), (200, 200))
#                for file_name in batch_x]), np.array(batch_y)


class ESC50(object):
    """This class is shipped with a generator yielding audio from ESC10 or
    ESC50. You may specify the folds you want to used

    eg:
    train = ESC50(folds=[1,2,3])
    train.data_gen.next()

    Parameters
    ----------
    folds : list of integers
        The folds you want to load

    only_ESC10 : boolean
        Wether to use ESC10 instead of ESC50
    """
    def __init__(self,
                 csv_path = '../meta/esc50.csv',
                 wav_dir = '../audio',
                 dest_dir = None,
                 only_ESC10=False,
                 folds=[1,2],
                 randomize=True,
                 audio_rate=44100,
                 strongAugment=False,
                 pad=0,
                 inputLength=0,
                 random_crop=False,
                 mix=False,
                 normalize=False):
        '''Initialize the generator

        Parameters
        ----------
        csv_path : str
            Path of the CSV file
        wav_dir : str
            path of the wav files
        dest_dir : str
            Directory where the sub-sampled wav are stored
        only_ESC10: Bool
            Wether to use ESC10 instead of ESC50
        randomize: Bool
            Randomize samples 
        audio_rate: int
            Audio rate of our samples
        strongAugment: Bool 
           rAndom scale and put gain in audio input 
        pad: int
            Add padding before and after audio signal
        inputLength: float
            Time in seconds of the audio input
        random_crop: Bool
            Perform random crops
        normalize: int
            Value used to normalize input
        mix: Bool
            Wether to mix samples or not (between classes learning)
        '''
        self.csv_path = csv_path
        self.wav_dir = wav_dir
        self.dest_dir = (dest_dir if dest_dir 
                                  else os.path.join(wav_dir, str(audio_rate)))
        self.audio_rate = audio_rate
        self.randomize = randomize
        self.audio_rate = audio_rate
        self.strongAugment = strongAugment
        self.pad = pad 
        self.inputLength = inputLength
        self.random_crop = random_crop
        self.normalize = normalize
        self.mix = mix
        self.n_classes = 50

        self.df = pd.read_csv(self.csv_path)
        self.df[self.df.fold.isin(folds)]
        if only_ESC10 is True:
            self.df = self.df[self.df['esc10']] 
            self.n_classes = 10

        self._preprocess_setup()
        self.data_gen = self._data_gen()

    @threadsafe_generator
    def _data_gen(self):
        self.stop = False
        while not self.stop:
            idxs1 = list(self.df.index)
            idxs2 = list(self.df.index)
            if self.randomize:
                random.shuffle(idxs1)
                random.shuffle(idxs2)

            for idx1, idx2 in zip(idxs1, idxs2):
                fname1 = self.df.filename[idx1]
                fname2 = self.df.filename[idx2]
                sound1 = self.fname_to_wav(fname1)
                sound2 = self.fname_to_wav(fname2)
                sound1 = self.preprocess(sound1)
                sound2 = self.preprocess(sound2)
                label1 = self.df.target[idx1]
                label2 = self.df.target[idx2]
                if self.n_classes == 10:
                    lbl_indexes = [0, 1, 10, 11, 12, 20, 21, 38, 40, 41]
                    label1 = lbl_indexes.index(label1)
                    label2 = lbl_indexes.index(label2)

                if self.mix:  # Mix two examples
                    r = np.array(random.random())
                    sound = U.mix(sound1, sound2, r, self.audio_rate)
                    sound = sound.astype(np.float32)
                    eye = np.eye(self.n_classes)
                    label = (eye[label1] * r + eye[label2] * (1 - r))
                    label = label.astype(np.float32)

                else:
                    sound, label = sound1, label1

                if self.strongAugment:
                    sound = U.random_gain(6)(sound).astype(np.float32)

                sound = sound[:, np.newaxis]

                yield sound, label

    @threadsafe_generator
    def batch_gen(self, batch_size):
        '''Generator yielding batches
        '''
        self.stop = False
        sounds = None
        labels = None
        data = self._data_gen()
        while not self.stop:
            for i in range(batch_size):
                if sounds is None:  # Initialize batch size
                    sound, label = next(data)
                    sounds = np.ndarray((batch_size,) + sound.shape)
                    sounds[0] = sound
                    if type(label) is int:
                        label = to_categorical(label, self.n_classes)
                    labels = np.ndarray((batch_size,) + label.shape)
                    labels[0] = label
                else:
                    sound, label = next(data)
                    if type(label) is int:
                        label = to_categorical(label, self.n_classes)
                    sounds[i] = sound
                    labels[i] = label

            sounds.reshape
            yield (sounds, labels)

    def fname_to_wav(self, fname):
        """Retrive wav data from fname
        """
        U.change_audio_rate(fname, self.wav_dir, self.audio_rate, self.dest_dir)
        fpath = os.path.join(self.dest_dir, fname)
        wav_freq, wav_data = sci_wav.read(fpath)
        return wav_data

    def _preprocess_setup(self):
        """Apply desired pre_processing to the input
        """
        self.preprocess_funcs = []
        if self.strongAugment:
            self.preprocess_funcs.append(U.random_scale(1.25))

        if self.pad > 0:
            self.preprocess_funcs.append(U.padding(self.pad))
        
        if self.random_crop:
            self.preprocess_funcs.append(
                U.random_crop(int(self.inputLength * self.audio_rate)))

        if self.normalize is True:
            self.preprocess_funcs.append(U.normalize(32768.0))

    def preprocess(self, audio):
        """Apply desired pre_processing to the input

        Parameters
        ----------
        audio: array 
            audio signal to be preprocess
        """
        for f in self.preprocess_funcs:
            audio = f(audio)

        return audio

    def __len__(self):
        return len(self.df)

    def get_example(self):
        if self.mix:  # Training phase of BC learning
            # Select two training examples
            while True:
                sound1, label1 = self.base[random.randint(0, len(self.base) -1)]
                sound2, label2 = self.base[random.randint(0, len(self.base) -1)]
                if label1 != label2:
                    break
            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)

            # Mix two examples
            r = np.array(random.random())
            sound = U.mix(sound1, sound2, r, self.opt.fs).astype(np.float32)
            eye = np.eye(self.opt.nClasses)
            label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)

        else:  # Training phase of standard learning or testing phase
            sound, label = self.base[i]
            sound = self.preprocess(sound).astype(np.float32)
            label = np.array(label, dtype=np.int32)

        if self.train and self.opt.strongAugment:
            sound = U.random_gain(6)(sound).astype(np.float32)

        return sound, label


def get_train_test(test_split=1, only_ESC10=False):
    '''return train and test depending on desired split
    '''
    train_splits = list(range(1,6))
    train_splits.remove(test_split)

    shared_params = {'audio_rate': 44100,
                     'only_ESC10': only_ESC10,
                     'pad': 0,
                     'normalize': True}

    train = ESC50(folds=train_splits,
                  randomize=True,
                  strongAugment=True,
                  random_crop=True,
                  inputLength=2,
                  mix=True,
                  **shared_params)

    test = ESC50(folds=[test_split],
                 randomize=False,
                 strongAugment=False,
                 random_crop=False,
                 inputLength=4,
                 mix=False,
                 **shared_params)
    
    return train, test


def test_plot_audio():
    '''Show a train and test split
    '''
    import matplotlib.pyplot as plt
    train, test = get_train_test(test_split=1)

    fig, axs = plt.subplots(10, 1)
    for i in range(10):
        sound, lbl = next(train.data_gen)
        print(sound)
        axs[i].plot(sound)

    fig, axs = plt.subplots(10, 1)
    for i in range(10):
        sound, lbl = next(test.data_gen)
        print(sound)
        axs[i].plot(sound)

    plt.show()


if __name__ == '__main__':
    test_plot_audio()
