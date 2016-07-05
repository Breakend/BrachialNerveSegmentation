from keras.callbacks import Callback
import numpy as np

class SimpleLrReducer(Callback):
    def __init__(self, patience=2, reduce_rate=0.94):
        super(Callback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.reduce_rate = np.float32(reduce_rate)

    def on_epoch_end(self, epoch, logs={}):
        if self.wait >= self.patience:
            self.wait = 0
            lr = self.model.optimizer.lr.get_value()
            self.model.optimizer.lr.set_value(lr*self.reduce_rate)
        self.wait += 1
