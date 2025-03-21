# L1 and L2 are two common loss functions in machine learning which are mainly used to minimize the error.

# L1 loss function are also known as Least Absolute Deviations in short LAD. L2 loss function are also known as Least square errors in short LS.

# Let's get brief of these two
# L1 Loss function
# It is used to minimize the error which is the sum of all the absolute differences in between the true value and the predicted value.
# L2 Loss Function
# It is also used to minimize the error which is the sum of all the squared differences in between the true value and the pedicted value.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
x_guess = tf.lin_space(-1., 1., 100)
x_actual = tf.constant(0,dtype=tf.float32)
with tf.Session() as sess:
    x_,l1_,l2_ = sess.run([x_guess, l1_loss, l2_loss])
    plt.plot(x_,l1_,label='l1_loss')
    plt.plot(x_,l2_,label='l2_loss')
    plt.legend()
    plt.show()

# Huber loss
x_guess2 = tf.linspace(-3.,5.,500)
x_actual2 = tf.convert_to_tensor([1.]*500)

#Hinge loss
#hinge_loss = tf.losses.hinge_loss(labels=x_actual2, logits=x_guess2)
hinge_loss = tf.maximum(0.,1.-(x_guess2*x_actual2))
0with tf.Session() as sess:
    x_,hin_ = sess.run([x_guess2, hinge_loss])
    plt.plot(x_,hin_,'--', label='hin_')
    plt.legend()
    plt.show()


