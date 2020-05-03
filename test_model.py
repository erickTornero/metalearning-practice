from hw1 import MANN
import tensorflow as tf
import numpy as np
import random
from load_data import DataGenerator
from tensorflow.python.platform import flags
from IPython.core.debugger import set_trace
FLAGS = flags.FLAGS

def test():
    ims = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, 784))
    labels = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))


    data_generator = DataGenerator(FLAGS.num_classes, FLAGS.num_samples + 1)
    o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
    o.load_weights('chpt_1700.ckpt')
    print('recovery data')
    out = o(ims, labels)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        #set_trace()
        i, l = data_generator.sample_batch('train', 100)
        feed = {ims: i.astype(np.float32),
                labels: l.astype(np.float32)}
        pred = sess.run([out], feed)
        pred    =   pred[0]
        #pred = pred.reshape(
        #        -1, FLAGS.num_samples + 1,
        #        FLAGS.num_classes, FLAGS.num_classes)
        pred = pred[:, -1, :, :].argmax(2)
        l = l[:, -1, :, :].argmax(2)
        print("Test Accuracy", (1.0 * (pred == l)).mean())


if __name__ == "__main__":
    test()