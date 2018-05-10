#!/usr/bin/env python3

"""
    Create tf.data.Dataset to iterate over JSON file

    Calculate sparsity defined as:

          sparsity = zero_elements / total_elements
                   = 1 - non_zero_elements / total_elements

    On top of that, profiles the memory consumption

    Usage:

          mprof run -T 2 read_json.py --size_of_dataset 2089
"""

import json
import numpy as np
import os
import tensorflow as tf
from absl import flags


flags.DEFINE_integer("size_of_dataset",
                     default=1000,
                     help="Size of the split datasets")
FLAGS = flags.FLAGS


total_elements = 61200


def dict_to_sparse_tensor(line):
    """ Convert dictionary to sparse tensor

        Initially, the shape of the sparse tensor
        is equal to the number of crystals [61200].

        Then, the sparse tensor is reshaped to [1, 170, 360]
    """
    d = json.loads(line)
    indices, values = zip(*[(int(k), float(v)) for k, v in d.items()])

    st = tf.SparseTensor(indices=np.array(indices)[:, np.newaxis],
                         values=values,
                         dense_shape=[total_elements])

    return tf.sparse_reshape(st, shape=[1, 170, 360])


def create_dataset(sublist):
    """ Create dataset from sublist
    """
    content = [dict_to_sparse_tensor(x) for x in sublist]
    dataset = tf.sparse_concat(axis=0, sp_inputs=content)
    dataset = tf.data.Dataset.from_tensor_slices((dataset))
    return dataset


def split_json(json_file, n):
    with open(json_file, 'r') as f:
        content = f.readlines()
    """ Create successive n-sized datasets
    """
    m = len(content)
    if n < m:
        sublists = [content[i:i+n] for i in range(0, m, n)]
        datasets = map(create_dataset, sublists)
        return datasets
    return map(create_dataset, [content])


@profile
def calculate_sparsity(datasets, n):
    with open(f'sparsity_{n}.txt', 'w') as out:
        for dataset in datasets:
            """ Creates an iterator
                Calculate the sparsity
            """
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            with tf.Session() as sess:
                while True:
                    try:
                        image = sess.run(next_element)
                        non_zero_elements = len(image.values)
                        sparsity = 1 - non_zero_elements/total_elements
                        print('{0:6f}'.format(sparsity), file=out)
                    except tf.errors.OutOfRangeError:
                        break


def main(argv):
    del argv  # unused
    path = '/home/jruizvar/private/work/jet-images/data'
    name = 'eminus_Ele-Eta0-PhiPiOver2-Energy50.json'
    json_file = os.path.join(path, name)

    """ Create datasets of a given size
    """
    size_of_dataset = FLAGS.size_of_dataset
    assert size_of_dataset > 0, f'{size_of_dataset} should be greater than 0'
    datasets = split_json(json_file, size_of_dataset)

    """ Iterate over datasets and calculate the sparsity.
        The results are saved in an output file
    """
    calculate_sparsity(datasets, size_of_dataset)


if __name__ == '__main__':
    tf.app.run(main)
