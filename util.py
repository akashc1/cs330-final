import numpy as np
import sys
import tensorflow as tf



def cross_entropy_loss(pred, label, weights=None):

    if weights is None:
      losses = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf.stop_gradient(label)) # (b, n)
      return tf.reduce_mean(losses)
    
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=tf.stop_gradient(label))
    return tf.multiply(losses, weights)


def accuracy(labels, predictions):
  return tf.reduce_mean(tf.cast(tf.equal(labels, predictions), dtype=tf.float32))