import json
import tensorflow as tf
from tensorflow import keras

class SWAGCallback(keras.callbacks.Callback):
    """Save SWAG mean and variance

    Arguments:
      output_prefix
      rank
    """
    def __init__(self, output_prefix, start=160, rank=139):
        # super(EarlyStoppingAtMinLoss, self).__init__()
        self.mean = 2000*[0.]
        # best_weights to store the weights at which the minimum loss occurs.
        self.moments = 2000*[0.]
        self.rank = rank
        self.start = start
        self.output_prefix = output_prefix
        self.cols = []

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start:
            self.mean = [((epoch-self.start)*prev + curr)/(epoch-self.start+1) for prev, curr in zip(self.mean, self.model.get_weights())]
            self.moments = [((epoch-self.start)*prev + curr**2)/(epoch-self.start+1) for prev, mean, curr in zip(self.moments, self.mean, self.model.get_weights())]
            self.cols = self.cols + [[(curr-mean) for mean, curr in zip(self.mean, self.model.get_weights())]]
        if len(self.cols) > self.rank:
            self.cols = self.cols[1:]

    def on_train_end(self, logs=None):
        json.dump([p.tolist() for p in self.mean], open(self.output_prefix + '_mean.json', 'w+'))
        diag = [moment - mean**2 for moment, mean in zip(self.moments, self.mean)]
        json.dump([p.tolist() for p in diag], open(self.output_prefix + '_diag.json', 'w+'))
        json.dump([[p.tolist() for p in col] for col in self.cols], open(self.output_prefix + '_cols.json', 'w+'))
