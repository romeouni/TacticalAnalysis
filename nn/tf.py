import tensorflow as tf

gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print('GPUS: ',gpus)

