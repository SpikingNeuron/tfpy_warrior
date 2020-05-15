import numpy as np
import tensorflow as tf


def gen(start_point: int):
    data = np.arange(10, dtype=np.int32) + start_point
    for d in data:
        yield d
        

ds = tf.data.Dataset.from_generator(
    generator=gen,
    output_types=tf.int32,
    args=(100,)
)

for d in ds.as_numpy_iterator():
    print(d)
    
ds = tf.data.Dataset.from_generator(
    generator=gen,
    output_types=tf.int32,
    args=(1000,)
)

for d in ds.as_numpy_iterator():
    print(d)