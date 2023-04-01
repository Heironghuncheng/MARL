# coding=utf-8
from pprint import pprint

import tensorflow as tf
from numpy import iterable

x = tf.constant([[1.]])
for i in x:
    print("iterable", iterable(i))
    for j in i:
        t = tf.math.is_nan(j)
        print(bool(t == tf.constant(False)))
