import tensorflow as tf

def individual_l2_loss(ca, x, y):
    t = y - ca.classify(x)
    return tf.reduce_sum(t**2, [1, 2, 3]) / 2

def batch_l2_loss(ca, x, y):
      return tf.reduce_mean(individual_l2_loss(ca, x, y))