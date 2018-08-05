import tensorflow as tf
import sys
import os

#Imprting the RNN and CNN utiltites from CNN Module
from CNN_Module.utils.conv2d_utils import *
from CNN_Module.utils.rnn_utils import *

def calculate_total_loss(Z,Y,scope=None):
    '''
    DESCRIPTION:
        This function combines all the losses i.e the model loss
        + L2Regularization loss

        Also, it will filter the L2 loss based on the namescope
        from all_losses collection to give specific loss for
        each GPU.
        (Though the L2Loss will be same for both GPU, but both
        GPU need to calculate their complete loss for computing
        gradient, so its better to do it from their local copy
        of variables)
    USAGE:
        INPUT:
            Z       :the final layer's unnormalized output of model
            Y       :the actual target label to train model on
            scope   :the namescope of the current tower to compute
                        the tower specific loss from the local copy
                        of the parameter present on each tower.
        OUTPUT:
            total_cost: the total cost of the model, which will be
                        used to calculate the gradient
    '''
    with tf.name_scope('loss_calculation'):
        #A list to combine all losses
        all_loss_list=[]

        #Calculating the L2_loss(scalar)
        reg_loss_list=tf.get_collection('reg_losses',scope=scope)
        l2_reg_loss=0.0
        if not len(reg_loss_list)==0:
            l2_reg_loss=tf.add_n(reg_loss_list,name='l2_reg_loss')
            tf.summary.scalar('l2_reg_loss',l2_reg_loss)
        all_loss_list.append(l2_reg_loss)

        #Calculating the model loss
        #Calculating the regression loss(scalar)
        regression_len=4
        regression_output=Z[:,0:regression_len]#0,1,2,3
        regression_label=Y[:,0:regression_len]#TRUTH

        #mean_squared_error Regrassion Loss
        regression_loss=tf.losses.mean_squared_error(regression_output,
                                    regression_label)

        #Defining new loss based on the Mean Percentage Error
        # absolute_diff=tf.losses.absolute_difference(regression_output,
        #                                             regression_label,
        #                                 reduction=tf.losses.Reduction.NONE)
        # regression_loss=tf.reduce_mean(tf.abs(tf.divide(
        #                         absolute_diff,regression_label+1e-10)))*100

        tf.summary.scalar('regression_loss',regression_loss)
        all_loss_list.append(regression_loss)

        #Calculating the x-entropy loss(scalar)
        class_output=Z[:,regression_len:]#4,....
        class_label=Y[:,regression_len:]#TRUTH
        categorical_loss=tf.losses.softmax_cross_entropy(class_label,
                                                        class_output)
        tf.summary.scalar('categorical_loss',categorical_loss)
        all_loss_list.append(categorical_loss)

        #calculating the total loss
        total_loss=tf.add_n(all_loss_list,name='merge_loss')
        tf.summary.scalar('total_loss',total_loss)

    return total_loss

def _conv2d_function_handle(X_img,is_training,iter_i,iter_end,reg_loss,tensor_array):
    '''
    DESCRIPTION:
        This will be a 2d convolution handle to be applied to all the
        detector-layers separately to genarate a sequence of output which
        will then be fed to the RNN block for inter-detector-layer
        connection.

        This convolution will generate a vector encoding of the detector
        layer since current RNN module just support the vector sequence
        handling. Later the CNN sequence handling will be added to have
        the sequence flow between the image layers (which still contain
        the spatial information).
    USAGE:
        INPUT:
            X_layer     : the hit-"image" of a particular layer of detector
            is_training : the training flag which will be used by
                            batchnormalization and the dropoutlayers.
        Output:
            Zx          : the output encoding of this image to be collected
                            and fed to RNN sequence.

        One drawback to this vectored approach is that model wont be able to
        share the low level features between the image layers and the
        features which depend on the inter-layer connection like for particle
        classification.
    '''
    #Model Hyperparameter
    bn_decision=False
    lambd=0.00
    dropout_rate=0.0

    #Running the convolution on each layers of the detector one by one
    #Slicing the input image for getting a detector layer
    X_layer=tf.expand_dims(X_img[:,:,:,iter_i],axis=-1,name='channel_dim')

    #Starting the model definition
    #First the convolution
    A1=rectified_conv2d(X_layer,
                        name='conv2d1',
                        filter_shape=(3,3),
                        output_channel=10,
                        stride=(1,1),
                        padding_type='VALID',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    #Then maxpooling the output
    A1Mp=max_pooling2d(A1,
                        name='mpool1',
                        filter_shape=(3,3),
                        stride=(2,2),
                        padding_type='VALID')

    #Definign the second layer
    #First the convolution
    A2=rectified_conv2d(A1Mp,
                        name='conv2d2',
                        filter_shape=(3,3),
                        output_channel=20,
                        stride=(1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    #Then maxpooling the output
    A2Mp=max_pooling2d(A2,
                        name='mpool2',
                        filter_shape=(3,3),
                        stride=(2,2),
                        padding_type='VALID')

    #Defining the third layer
    #First the convolution
    A3=rectified_conv2d(A2Mp,
                        name='conv2d3',
                        filter_shape=(3,3),
                        output_channel=30,
                        stride=(1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    #Then maxpooling the output
    A3Mp=max_pooling2d(A3,
                        name='mpool3',
                        filter_shape=(3,3),
                        stride=(2,2),
                        padding_type='VALID')

    #Defining the fourth layer
    #First the convolution
    A4=rectified_conv2d(A3Mp,
                        name='conv2d4',
                        filter_shape=(3,3),
                        output_channel=40,
                        stride=(1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    #Then maxpooling the output
    A4Mp=max_pooling2d(A4,
                        name='mpool4',
                        filter_shape=(3,3),
                        stride=(2,2),
                        padding_type='VALID')

    #Defining the fifth layer
    #First the convolution
    A5=rectified_conv2d(A4Mp,
                        name='conv2d5',
                        filter_shape=(3,3),
                        output_channel=50,
                        stride=(1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    #Then maxpooling the output
    A5Mp=max_pooling2d(A5,
                        name='mpool5',
                        filter_shape=(3,3),
                        stride=(2,2),
                        padding_type='VALID')

    #Defining the sixth layer
    #First defining the convolution
    A6=rectified_conv2d(A5Mp,
                        name='conv2d6',
                        filter_shape=(3,3),
                        output_channel=60,
                        stride=(1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    #Then maxpooling the output
    A6Mp=max_pooling2d(A6,
                        name='mpool6',
                        filter_shape=(3,3),
                        stride=(2,2),
                        padding_type='VALID')

    #now it's time to flatten out all the activation
    Z7=simple_fully_connected(A6Mp,
                                name='fc1',
                                output_dim=1000,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd,
                                flatten_first=True,
                                apply_relu=False)

    #Setting the final activation to the layer output
    det_layer_activation=Z7


    ##################### CUSTOM OPS (Directly copy them)#############
    ##################### No need to change with models ###############
    #adding the output encoding to the tensorarray
    tensor_array=tensor_array.write(iter_i,det_layer_activation)

    #Updating the iter_i
    iter_i=iter_i+1

    #Calculating the regularization loss of the conv2d layer for passing it
    #out of the tf.while_loop (though doing it each time it should remain same)
    #Retreiving the collection in current scope and geting the collection
    reg_loss_list=tf.get_collection('all_losses',
                            scope=tf.contrib.framework.get_name_scope())
    #print tf.contrib.framework.get_name_scope()
    l2_reg_loss_conv=0.0
    if not len(reg_loss_list)==0:
        l2_reg_loss_conv=tf.add_n(reg_loss_list,name='l2_reg_loss_conv')
        #tf.summary.scalar('l2_reg_loss_conv',l2_reg_loss)
    #Assigning this l2Loss of current conv layer to the reg_loss loop_var
    reg_loss=l2_reg_loss_conv

    #Setting the variable scope to True
    tf.get_variable_scope().reuse_variables()

    #returning all the loop_vars
    return [X_img,is_training,iter_i,iter_end,reg_loss,tensor_array]

def model8(X_img,is_training):
    '''
    DESCRIPTION:
        In this model we will test the performance of the RNN module
        on the detector-hits.
    USAGE:
        INPUT:
            X_img       : the complete 3d hit-image of the detector
            is_training : the training flag which will be used by the
                            batchnorm and dropout layers
        OUTPUT:
            Z_out       : the final unnormalized output of the model
    '''
    #Rnn Hyperparameters
    rnn_lambd=0.0

    #Now invoking the RNN block to create create the RNN layers along
    #with the required convolution using the CNN function handle
    Z_list=simple_vector_RNN_block(X_img,
                                    is_training,
                                    _conv2d_function_handle,
                                    num_of_sequence_layers=1,
                                    hidden_state_dim_list=[1000],
                                    output_dimension_list=[6],
                                    output_type='vector',
                                    output_norm_list=[None],
                                    num_detector_layers=40,
                                    weight_decay=rnn_lambd)

    #Returning the unnormalized output of the whole model
    Z_out=Z_list[0]
    return Z_out

def model9(X_img,is_training):
    '''
    DESCRIPTION:
        In this model we will try to test the reason for bad learning of
        model8, and confirm if it is caused by the vanishing gradient
        problem or not by haing more direct channel for the gradient
        propagation in each detector-layer
    USAGE:
        as the above model
    '''
    #RNN Hyperparameter
    rnn_lambd=0.0
    bn_decision=False
    fc_lambd=0.0
    dropout_rate=0.0

    #Invoking the RNN block which will implicitely create the conv2d
    #activation first before adding the RNN layer to the vector-encoding
    Z_list=simple_vector_RNN_block(X_img,
                                    is_training,
                                    _conv2d_function_handle,
                                    num_of_sequence_layers=1,
                                    hidden_state_dim_list=[1000],
                                    output_dimension_list=[20],
                                    output_type='sequence',
                                    output_norm_list=['relu'],
                                    num_detector_layers=40,
                                    weight_decay=rnn_lambd)

    #Now we will be aggregating the output of the sequence layer
    Z_aggregated=tf.concat(Z_list,axis=1,name='seq_concat')

    #Now we will be reducing the aggregation by FC layer
    Z_out=simple_fully_connected(Z_aggregated,
                                name='fc_aggregated',
                                output_dim=6,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=fc_lambd,
                                apply_relu=False)

    #Now finally returning this unnormalized vector as output of model
    return Z_out
