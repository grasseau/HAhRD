import tensorflow as tf
from conv2d_utils import *
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#Global Parameters
n_classes=10
batch_size=100

def make_model():
    X=tf.placeholder(tf.float32,[None,784])

    A1=simple_fully_connected(X,'fc1',100,None,apply_batchnorm=False)
    A2=simple_fully_connected(A1,'fc2',100,None,apply_batchnorm=False)
    A3=simple_fully_connected(A2,'fc3',100,None,apply_batchnorm=False)

    Z4=simple_fully_connected(A3,'fc4',10,None,apply_batchnorm=False,
                                apply_relu=False)

    return Z4

def train_net():
    prediction=make_model()
    Y=tf.placeholder(tf.float32,[None,10])
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                                labels=Y))
    optimizer=tf.train.AdamOptimizer().minimize(cost)

    epochs=10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss=0
            for _ in range(mnist.train.num_examples/batch_size):
                epochs_x,epoch_y=mninst.train._next_batch(batch_size)
                epochs_x=tf.transpose(epochs_x)
                epochs_y=tf.transpose(epochs_y)
                _,c=sess.run([optimizer,cost],feed_dict={X:epoch_x,y:epoch_y})
                epoch_loss +=c

            print 'Epoch ',epoch,' out of ',epochs,'loss: ',epoch_loss

            correct=tf.equal(tf.argmax(prediction),tf.argmax(Y))
            accuracy=tf.reduce_mean(tf.cast(correct,'float'))
            print 'Accuracy: ',accuracy.eval({X:mnist.test.images,Y:mnist.test.labels})

train_net()
