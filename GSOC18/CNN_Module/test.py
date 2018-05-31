import tensorflow as tf
from conv2d_utils import *
from conv3d_utils import *
import datetime

#Getting the training data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

################# Global Parameters  ####################
n_classes=10
batch_size=128
epochs=5
summary_filename="tmp/mnist/3"

##################  Global Placeholders ##################
X=tf.placeholder(tf.float32,[None,784],name='X')
Y=tf.placeholder(tf.float32,[None,10],name='Y')
is_training=tf.placeholder(tf.bool,[],name='training_flag')


################### HELPER FUNCTION #######################
def add_summary(object):
    tf.summary.histogram(object.op.name,object)

def add_all_trainiable_var_summary():
    for var in tf.trainable_variables():
        add_summary(var)

def calculate_total_loss(prediction,Y):
    #Calculating the cross entropy loss from predicion and labels
    x_entropy_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    logits=prediction,labels=Y))
    tf.summary.scalar('x_entropy_loss',x_entropy_loss)

    #Calculating the L-2 regularization Loss
    reg_loss_list=tf.get_collection('all_losses')
    l2_reg_loss=0.0
    if not len(reg_loss_list)==0:
        l2_reg_loss=tf.add_n(reg_loss_list,name='l2_reg_loss')
        tf.summary.scalar('l2_reg_loss',l2_reg_loss)

    #Now adding up all the losses
    total_cost=tf.add(l2_reg_loss,x_entropy_loss,name='merge_loss')

    return total_cost

def get_optimizer_op(total_cost,optimizer_type=tf.train.AdamOptimizer()):
    '''
    DECRIPTION:
        This function will add the extra dependency of updating the
        moving averages of beta and gamma of Batchnormalization if any
        to the optimizer.
    USAGE:
        INPUT:
            total_cost      : the total loss of the whole model
            optimizer_type  : a function reference of the optimizer
        OUTPUT:
            optimizer       : the handle of optimizer op to run the gradient
                               descent
    '''
    #For updating the moving averages of the batch norm parameters
    extra_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        #Creating the optimizer op node in the graph
        optimizer=optimizer_type.minimize(total_cost)

    return optimizer

################ MODEL DEFINITION ########################
def make_model_linear():
    lambd=0
    bn_decision=True
    A1=simple_fully_connected(X,'fc1',50,is_training,dropout_rate=0.4,
                            apply_batchnorm=bn_decision,weight_decay=lambd,
                            apply_relu=True)
    add_summary(A1)

    # A2=simple_fully_connected(A1,'fc2',20,is_training,apply_batchnorm=bn_decision,weight_decay=lambd)
    # add_summary(A2)

    #A3=simple_fully_connected(A2,'fc3',100,is_training,apply_batchnorm=bn_decision,weight_decay=lambd)

    Z4=simple_fully_connected(A1,'fc4',10,is_training,apply_batchnorm=bn_decision,weight_decay=lambd,
                                apply_relu=False)
    add_summary(Z4)

    return Z4

def make_model_conv():
    #Current Hyperparameter (will be separated later)
    bn_decision=False
    lambd=0.0
    dropout_rate=0.0

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
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A1Mp=max_pooling2d(A1,name='mpool1',
                        filter_shape=(3,3),stride=(1,1),
                        padding_type='VALID')

    #The first Convolutional layer
    # A2=rectified_conv2d(A1Mp,
    #                     name='conv2',
    #                     filter_shape=(5,5),
    #                     output_channel=10,
    #                     stride=(1,1),
    #                     padding_type='SAME',
    #                     is_training=is_training,
    #                     dropout_rate=dropout_rate,
    #                     apply_batchnorm=bn_decision,
    #                     weight_decay=lambd,
    #                     apply_relu=True)
    # A2Mp=max_pooling2d(A2,name='mpool2',
    #                     filter_shape=(5,5),stride=(1,1),
    #                     padding_type='VALID')

    #Adding an identity block
    A3=identity_residual_block(A1Mp,'identity_block',
                                num_channels=[3,3,5],#here last channel is fixed since it has to be equal to input
                                mid_filter_shape=(5,5),
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd)
    # A3Mp=max_pooling2d(A3,name='mpool3',
    #                     filter_shape=(5,5),stride=(1,1),
    #                     padding_type='VALID')

    # A4=convolutional_residual_block(A1Mp,
    #                                 name='conv_res_block',
    #                                 num_channels=[3,5,12],#here last channel can be diff from input
    #                                 first_filter_stride=(2,2),
    #                                 mid_filter_shape=(5,5),
    #                                 is_training=is_training,
    #                                 dropout_rate=dropout_rate,
    #                                 apply_batchnorm=bn_decision,
    #                                 weight_decay=lambd)

    A5=inception_block(A3,
                        name='inception1',
                        final_channel_list=[3,2,1,2],
                        compress_channel_list=[2,2],
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd)

    A6=simple_fully_connected(A5,name='fc1',output_dim=25,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                flatten_first=True,weight_decay=lambd)

    Z5=simple_fully_connected(A6,name='fc2',output_dim=10,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd,apply_relu=False)

    return Z5

def make_model_conv3d():
    bn_decision=False
    lambd=0.0
    dropout_rate=0.0

    #Converting it into a five dimensional tensor
    X_img=tf.reshape(X,shape=[-1,28,28,1,1])

    #Adding the first convolution layer
    A1=rectified_conv3d(X_img,
                        name='conv3d1',
                        filter_shape=(3,3,1),
                        output_channel=5,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A1Mp=max_pooling3d(A1,
                        name='mpool1',
                        filter_shape=(5,5,1),
                        stride=(1,1,1),
                        padding_type='VALID')

    # A2=identity3d_residual_block(A1Mp,
    #                             name='identity1',
    #                             num_channels=[2,2,5],
    #                             mid_filter_shape=(5,5,1),
    #                             is_training=is_training,
    #                             dropout_rate=dropout_rate,
    #                             apply_batchnorm=bn_decision,
    #                             weight_decay=lambd)

    # A2=convolutional3d_residual_block(A1Mp,
    #                                   name='conv_res_block',
    #                                   num_channels=(2,2,2),
    #                                   first_filter_stride=(2,2,1),
    #                                   mid_filter_shape=(5,5,1),
    #                                   is_training=is_training,
    #                                   dropout_rate=dropout_rate,
    #                                   apply_batchnorm=bn_decision,
    #                                   weight_decay=lambd)

    A2=inception3d_block(A1Mp,
                        name='inception1',
                        final_channel_list=[3,2,1,1],
                        compress_channel_list=(1,1),
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd)



    Z5=simple_fully_connected(A2,name='fc2',output_dim=10,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd,
                                flatten_first=True,apply_relu=False)
    return Z5

######################## Main Training Function ######################
def train_net(prediction):
    #Calculating cost and optimizer op
    total_cost=calculate_total_loss(prediction,Y)
    optimizer=get_optimizer_op(total_cost,
                        optimizer_type=tf.train.AdamOptimizer())

    #Starting the session to run the graph
    with tf.Session() as sess:
        #Initializing all the variables
        sess.run(tf.global_variables_initializer())

        #Getting the File writer to add the summary
        merged_summary=tf.summary.merge_all()
        writer=tf.summary.FileWriter(summary_filename)
        writer.add_graph(sess.graph)

        for epoch in range(epochs):
            t0=datetime.datetime.now()
            epoch_loss=0
            for _ in range(mnist.train.num_examples/batch_size):
                #Running the training step
                epochs_x,epochs_y=mnist.train.next_batch(batch_size)
                _,c=sess.run([optimizer,total_cost],feed_dict={X:epochs_x,Y:epochs_y,is_training:True})
                epoch_loss +=c

            #Writing the merged summary with the test data
            s=sess.run(merged_summary,feed_dict={X:mnist.test.images,Y:mnist.test.labels,is_training:False})
            writer.add_summary(s,epoch)

            #Printing the loss on the currrent training loss
            t1=datetime.datetime.now()
            print 'Epoch ',epoch,' out of ',epochs,'loss: ',epoch_loss,' in ',t1-t0

        #BEWARE: we have to use {is_training = False} while doing inference
        correct=tf.equal(tf.argmax(prediction,axis=1),tf.argmax(Y,axis=1))
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        #tf.summary.scalar('test_accuracy',accuracy)
        acc_val=accuracy.eval({X:mnist.test.images,Y:mnist.test.labels,is_training:False})
        #writer.add_summary(acc_val,epoch)
        print 'Accuracy on test set : ',acc_val,'\n'

        #Closing the writer
        writer.close()


################### Main Calling Function ##################
if __name__=="__main__":
    #Creating the graph
    prediction=make_model_conv3d()

    #Training the graph
    train_net(prediction)
