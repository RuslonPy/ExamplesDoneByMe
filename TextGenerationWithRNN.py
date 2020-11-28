import tensorflow as tf
import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print("Leanght of the text: {}".format(len(text)))

