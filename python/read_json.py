"""
    Deploy tf.data.Dataset to iterate over JSON file
"""

import json
import numpy as np
import os
import tensorflow as tf


def dict_to_sparse_tensor(line):
    """ Convert dictionary to sparse tensor

        Initially, the shape of the sparse tensor
        is equal to the number of crystals (61200).

        Then, the sparse tensor is reshaped to [1, 170, 360]
    """
    d = json.loads(line)
    indices, values = zip(*[(int(k), float(v)) for k, v in d.items()])
    st = tf.SparseTensor(indices=np.array(indices)[:, np.newaxis],
                         values=values,
                         dense_shape=[61200])
    return tf.sparse_reshape(st, shape=[1, 170, 360])


def create_dataset(json_file):
    """ Create dataset from JSON file
        taking care of null values
    """
    with open(json_file, 'r') as f:
        content = f.readlines()

    content = [x for x in content if x != 'null\n']
    content = [dict_to_sparse_tensor(x) for x in content]

    dataset = tf.sparse_concat(axis=0, sp_inputs=content)
    dataset = tf.data.Dataset.from_tensor_slices((dataset))
    return dataset


if __name__ == '__main__':
    path = '/Users/jose/Work/jet-images/data'
    name = 'eminus_Ele-Eta0-PhiPiOver2-Energy50.json'
    json_file = os.path.join(path, name)

    """ Create dataset and iterate over it
    """
    dataset = create_dataset(json_file)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        while True:
            try:
                value = sess.run(next_element)
                print(value.dense_shape)
            except tf.errors.OutOfRangeError:
                break
