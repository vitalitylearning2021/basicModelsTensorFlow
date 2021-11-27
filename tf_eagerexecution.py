# -*- coding: utf-8 -*-
"""TF_EagerExecution.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15OYPHT77FQI_7l8wuAe7JOic6oRWe9Ge
"""

import tensorflow as tf
import tensorboard

# --- Check if eager execution is enabled
tf.executing_eagerly()

print(tf.multiply(6, 7).numpy() == 42)

# --- Disable eager execution
tf.compat.v1.disable_eager_execution()

# --- Check if eager execution is enables
tf.executing_eagerly()

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
tf.compat.v1.reset_default_graph()

a = tf.Variable(5, name='variableA')
b = tf.Variable(6, name='variableB')
c = tf.multiply(a, b, name='Mul')

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

writer = tf.compat.v1.summary.FileWriter('./graphs', graph=sess.graph)

print(sess.run(c)) 

writer.flush()
writer.close()
# %tensorboard --logdir='./graphs'