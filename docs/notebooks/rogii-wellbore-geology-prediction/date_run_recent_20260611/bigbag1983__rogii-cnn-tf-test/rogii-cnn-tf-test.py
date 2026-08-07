import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
try:
    with tf.device('/GPU:0'):
        x = tf.random.normal([2, 64, 64, 3])
        y = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
        print("TF Conv2d OK:", y.shape)
except Exception as e:
    print("TF Conv2d FAILED:", e)
print("Done.")
