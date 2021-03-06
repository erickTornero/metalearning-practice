import numpy as np
import random
import tensorflow as tf
from load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras import layers
from IPython.core.debugger import set_trace
FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 16,
                     'Number of N-way classification tasks per batch')


def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """
    #set_trace()
    losses  =   []
    for iclass in  range(labels.shape[-1]):
        _predictions    =   preds[:, -1, iclass, :]
        _labels         =   labels[:, -1, iclass, :]
        loss            =   tf.losses.sigmoid_cross_entropy(_labels, _predictions)
        #tf.print(loss)
        losses.append(loss)
    finalloss   =   tf.add_n(losses)
    return finalloss
    #############################
    #### YOUR CODE GOES HERE ####
    
    #############################


class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        #set_trace()

        predicted_label =   []
        ## Concatenate input
        #input_labels[:, -1, :, :]   =   tf.math.scalar_mul(0.0, input_labels[:, -1, :, :])
        input_concatenated = tf.concat((input_images, input_labels), -1)
        
        for i_class in range(self.num_classes):
            x     =   self.layer1(input_concatenated[:,:,i_class,:])
            x     =   self.layer2(x)  
            predicted_label.append(x)
        #############################
        out =   tf.stack(predicted_label, axis=2)
        return out


def train():
    ims = tf.placeholder(tf.float32, shape=(
        None, FLAGS.num_samples + 1, FLAGS.num_classes, 784))
    labels = tf.placeholder(tf.float32, shape=(
        None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))

    data_generator = DataGenerator(
        FLAGS.num_classes, FLAGS.num_samples + 1)

    o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
    out = o(ims, labels)

    loss = loss_function(out, labels)
    optim = tf.train.AdamOptimizer(0.001)
    optimizer_step = optim.minimize(loss)
    checkpoint_path = 'chpt_{}.ckpt'


    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        for step in range(50000):
            i, l = data_generator.sample_batch('train', FLAGS.meta_batch_size)
            feed = {ims: i.astype(np.float32), labels: l.astype(np.float32)}
            _, ls = sess.run([optimizer_step, loss], feed)

            if step % 100 == 0:
                print("*" * 5 + "Iter " + str(step) + "*" * 5)
                i, l = data_generator.sample_batch('test', 100)
                feed = {ims: i.astype(np.float32),
                        labels: l.astype(np.float32)}
                pred, tls = sess.run([out, loss], feed)
                print("Train Loss:", ls, "Test Loss:", tls)
                pred = pred.reshape(
                    -1, FLAGS.num_samples + 1,
                    FLAGS.num_classes, FLAGS.num_classes)
                pred = pred[:, -1, :, :].argmax(2)
                l = l[:, -1, :, :].argmax(2)
                print("Test Accuracy", (1.0 * (pred == l)).mean())
                o.save_weights(checkpoint_path.format(step))

if __name__ == "__main__":
    train()