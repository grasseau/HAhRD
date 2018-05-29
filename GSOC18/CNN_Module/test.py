import tensorflow as tf
from conv2d_utils import *
import datetime


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#Global Parameters
n_classes=10
batch_size=128
epochs=10

#Global Placeholders
X=tf.placeholder(tf.float32,[None,784],name='X')
Y=tf.placeholder(tf.float32,[None,10],name='Y')
is_training=tf.placeholder(tf.bool,[],name='training_flag')

def make_model_linear():
    lambd=0.1
    bn_decision=True
    A1=simple_fully_connected(X,'fc1',100,is_training,apply_batchnorm=bn_decision,weight_decay=lambd)
    A2=simple_fully_connected(A1,'fc2',100,is_training,apply_batchnorm=bn_decision,weight_decay=lambd)
    A3=simple_fully_connected(A2,'fc3',100,is_training,apply_batchnorm=bn_decision,weight_decay=lambd)

    Z4=simple_fully_connected(A3,'fc4',10,is_training,apply_batchnorm=bn_decision,weight_decay=lambd,
                                apply_relu=False)

    return Z4

def make_model_conv():
    bn_decision=False

    X_img=tf.reshape(X,[-1,28,28,1])
    #with tf.device('/cpu:0'):
    #The first convolutional layer
    A1=rectified_conv2d(X_img,
                        name='conv1',
                        filter_shape=(3,3),
                        output_channel=5,
                        stride=(1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        apply_batchnorm=bn_decision,
                        weight_decay=None,
                        apply_relu=True)
    A1Mp=max_pooling2d(A1,name='mpool1',
                        filter_shape=(3,3),stride=(1,1),
                        padding_type='VALID')

    #The first Convolutional layer
    A2=rectified_conv2d(A1Mp,
                        name='conv2',
                        filter_shape=(5,5),
                        output_channel=10,
                        stride=(1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        apply_batchnorm=bn_decision,
                        weight_decay=None,
                        apply_relu=True)
    A2Mp=max_pooling2d(A2,name='mpool2',
                        filter_shape=(5,5),stride=(1,1),
                        padding_type='VALID')

    #The third convolution Layer
    #Adding an identity block
    A3=identity_residual_block(A2Mp,'identity_block',
                                num_channels=[3,5,10],#here last channel is fixed since it has to be equal to input
                                mid_filter_shape=(5,5),
                                is_training=is_training,
                                apply_batchnorm=bn_decision,
                                weight_decay=None)
    # A3Mp=max_pooling2d(A3,name='mpool3',
    #                     filter_shape=(5,5),stride=(1,1),
    #                     padding_type='VALID')

    # A4=convolutional_residual_block(A3,
    #                                 name='conv_res_block',
    #                                 num_channels=[3,5,12],#here last channel can be diff from input
    #                                 first_filter_stride=(2,2),
    #                                 mid_filter_shape=(5,5),
    #                                 is_training=is_training,
    #                                 apply_batchnorm=bn_decision,
    #                                 weight_decay=None)

    A4=inception_block(A3,
                        name='inception1',
                        final_channel_list=[3,2,1,2],
                        compress_channel_list=[2,2],
                        is_training=is_training,
                        apply_batchnorm=bn_decision,
                        weight_decay=None)

    A5=simple_fully_connected(A4,name='fc1',output_dim=25,
                                is_training=is_training,apply_batchnorm=bn_decision,
                                flatten_first=True,weight_decay=None)

    Z5=simple_fully_connected(A5,name='fc2',output_dim=10,
                                is_training=is_training,apply_batchnorm=bn_decision,
                                weight_decay=None,apply_relu=False)

    return Z5

def train_net():
    prediction=make_model_conv()

    extra_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #For updating the moving averages of the batch norm parameters
    with tf.control_dependencies(extra_update_ops):
        x_entropy_cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                                    labels=Y))
        tf.add_to_collection('all_losses',x_entropy_cost)

    #Adding all the losses (including from l2 regularization if any)
    all_losses=tf.get_collection('all_losses')
    total_cost=tf.add_n(all_losses,name='merge_loss')
    #Calling the optimizer to optimize the cost
    optimizer=tf.train.AdamOptimizer().minimize(total_cost)

    #Invoking the file writer to write out the summary
    writer=tf.summary.FileWriter("/tmp/mnist/1")

    #Starting the session to run the graph
    # config=tf.ConfigProto()
    # config.gpu_options.allow_growth=True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        for epoch in range(epochs):
            t0=datetime.datetime.now()
            epoch_loss=0
            for _ in range(mnist.train.num_examples/batch_size):
                epochs_x,epochs_y=mnist.train.next_batch(batch_size)
                # epochs_x=tf.transpose(epochs_x)
                # epochs_y=tf.transpose(epochs_y)
                _,c=sess.run([optimizer,total_cost],feed_dict={X:epochs_x,Y:epochs_y,is_training:True})
                epoch_loss +=c

            t1=datetime.datetime.now()
            print 'Epoch ',epoch,' out of ',epochs,'loss: ',epoch_loss,' in ',t1-t0

        #BEWARE: we have to use this while doing inference (is_training = False)
        #is_training=False
        correct=tf.equal(tf.argmax(prediction,axis=1),tf.argmax(Y,axis=1))
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        print 'Accuracy: ',accuracy.eval({X:mnist.test.images,Y:mnist.test.labels,is_training:False})##

        #writer.flush()
        writer.close()
train_net()
