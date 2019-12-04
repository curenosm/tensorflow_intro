import random
import math

import tensorflow as tf
import numpy as np
from tensorflow import keras

function_to_predict = lambda x: 79.3*x + 20.5
number_of_inputs = 1000
f_MIN = 0
f_MAX = 10000
number_of_epochs = 10000
number_to_test = 100
expected_result = function_to_predict(number_to_test)

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='rmsprop', loss='mean_squared_error')

x_list = [float(math.floor(random.random()*f_MAX+f_MIN)) for number in range(number_of_inputs)]
y_list = [function_to_predict(e) for e in x_list]

xs = np.array(x_list, dtype=float)
ys = np.array(y_list, dtype=float)

model.fit(xs, ys, epochs=number_of_epochs)
print(model.predict([number_to_test]))
print(expected_result)