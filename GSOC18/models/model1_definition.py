import tensorflow as tf
import sys
import os

#Appending the path  of the utility files
sys.path.append(
    os.path.join(os.path.dirname(sys.path[0]),
                    'CNN_Module/utils')
)

from conv2d_utils import *
from conv3d_utils import *

def calculate_model_accuracy(Z,Y):
    '''
    DESCRIPTION:
        This function will be used to calculate the model accuracy.
        Both regression accuracy and classification accuracy
        will be displayed separately in the tensorboard.
        This formulation of accuracy will be unique for each
        model description.
    USAGE:
        INPUT:
            Z   :the activation from the final layer of the model
            Y   :the target label of the model to aim at (Arjun)
        OUPUT:
            regression_accuracy     : 100- (% error)(WRONG, Think better metric)
            classification_accuracy : (correct/total)*100

            These output will be optionally caught by the caller
            if we want to display them in the terminal also and
            then run in the ops along with losses.
    '''
    with tf.name_scope('accuracy_calculation'):
        #Regression accuracy calcuation as 100-%error
        #Energy_accuracy/error
        energy_abs_diff=tf.losses.absolute_difference(Z[:,0],
                                    Y[:,0],reduction=tf.losses.Reduction.NONE)
        energy_error=(tf.reduce_mean(tf.divide(
                                energy_abs_diff,Y[:,0]+1e-10)))*100
        tf.summary.scalar('percentage_energy_error',energy_error)

        #posx Accuracy/Error
        posx_abs_diff=tf.losses.absolute_difference(Z[:,1],
                                    Y[:,1],reduction=tf.losses.Reduction.NONE)
        posx_error=(tf.reduce_mean(tf.abs(tf.divide(
                                    posx_abs_diff,Y[:,1]+1e-10))))*100
        tf.summary.scalar('percentage_posx_error',posx_error)

        #posy Accuracy/Error
        posy_abs_diff=tf.losses.absolute_difference(Z[:,2],
                                    Y[:,2],reduction=tf.losses.Reduction.NONE)
        posy_error=(tf.reduce_mean(tf.abs(tf.divide(
                                    posy_abs_diff,Y[:,2]+1e-10))))*100
        tf.summary.scalar('percentage_posy_error',posy_error)

        #posz Accuracy/Error
        posz_abs_diff=tf.losses.absolute_difference(Z[:,3],
                                    Y[:,3],reduction=tf.losses.Reduction.NONE)
        posz_error=(tf.reduce_mean(tf.abs(tf.divide(
                                    posz_abs_diff,Y[:,3]+1e-10))))*100
        tf.summary.scalar('percentage_posz_error',posz_error)


        #Calculation of Classification error
        regression_len=4
        prediction=tf.argmax(Z[:,regression_len:],axis=1)
        label=tf.argmax(Y[:,regression_len:],axis=1)

        correct=tf.equal(prediction,label)
        classification_accuracy=tf.reduce_mean(tf.cast(correct,'float'))*100
        tf.summary.scalar('percentage_classification_accuracy',classification_accuracy)

    #Returning the accuracy tuple to be used in inference module
    accuracy_tuple=(energy_error,
                    posx_error,
                    posy_error,
                    posz_error,
                    classification_accuracy)
    return accuracy_tuple


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
        reg_loss_list=tf.get_collection('all_losses',scope=scope)
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


def model1(X,is_training):
    '''
    DESCRIPTION:
        Defining the first CNN model for the HGCAL. This function will
        take the input X and apply the CNN as defined in this model
        and give the unrectified output of the last layer as output
        for the computing the cost/error for training.

    USAGE:
        INPUT:
            X           : the input "image" to the model
            is_training : the training flag to know whether we are in
                            training mode or not which is required
                            for batch norm and dropout layers to modify
                            thier behaviour accordingly.
        OUTPUT:
            Z           : the output (un rectified output) of this
                            model to furthur compute the loss and hence
                            gradient.
    '''
    #Model Hyperparameter Section
    #(A separate handling mechanism will be developed for this later)
    bn_decision=False           #Batch-Normalization (ON/OFF)
    lambd=0.0                   #control L2-Regularization of weights
    dropout_rate=0.0            #The dropout rates of activation in any layer


    #Reshaping the input to appropriate format
    X_img=tf.expand_dims(X,axis=-1,name='channel_dim')

    #Passing it through the first layer
    A1=rectified_conv3d(X_img,
                        name='conv3d1',
                        filter_shape=(3,3,3),
                        output_channel=5,
                        stride=(1,1,1),
                        padding_type='VALID',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)#it is default no need to write
    A1Mp=max_pooling3d(A1,
                        name='mpool1',
                        filter_shape=(3,3,3),
                        stride=(2,2,2),
                        padding_type='VALID')

    #Passing it through the second layer
    A2=rectified_conv3d(A1Mp,
                        name='conv3d2',
                        filter_shape=(3,3,3),
                        output_channel=10,
                        stride=(1,1,1),
                        padding_type='VALID',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)#it is default no need to write
    A2Mp=max_pooling3d(A2,
                        name='mpool2',
                        filter_shape=(3,3,3),
                        stride=(2,2,2),
                        padding_type='VALID')

    #Writing the third layer similar as the above pattern
    A3=rectified_conv3d(A2Mp,
                        name='conv3d3',
                        filter_shape=(3,3,3),
                        output_channel=10,
                        stride=(1,1,1),
                        padding_type='VALID',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)#it is default no need to write
    A3Mp=max_pooling3d(A3,
                        name='mpool3',
                        filter_shape=(3,3,3),
                        stride=(2,2,2),
                        padding_type='VALID')

    #Passing the activation to the fourth layer which is identity
    A4=identity3d_residual_block(A3Mp,
                                name='identity1',
                                num_channels=[2,2,10],
                                mid_filter_shape=(3,3,3),
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd)
    A4Mp=max_pooling3d(A4,
                        name='mpool4',
                        filter_shape=(3,3,3),
                        stride=(3,3,1),
                        padding_type='SAME')

    #Passing through the fifth layer
    A5=convolutional3d_residual_block(A4Mp,
                                        name='conv_res1',
                                        num_channels=[2,2,15],
                                        first_filter_stride=(1,1,1),
                                        mid_filter_shape=(3,3,3),
                                        is_training=is_training,
                                        dropout_rate=dropout_rate,
                                        apply_batchnorm=bn_decision,
                                        weight_decay=lambd)
    A5Mp=max_pooling3d(A5,
                        name='mpool5',
                        filter_shape=(3,3,3),
                        stride=(3,3,1),
                        padding_type='SAME')

    #Passing it through the sixth layer
    #Passing the activation to the fourth layer which is identity
    A6=identity3d_residual_block(A5Mp,
                                name='identity2',
                                num_channels=[2,2,15],
                                mid_filter_shape=(3,3,3),
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd)
    A6Mp=max_pooling3d(A6,
                        name='mpool6',
                        filter_shape=(3,3,3),
                        stride=(3,3,1),
                        padding_type='SAME')

    #Now finally flattening it and connecting it with fully
    #connected layer
    Z7=simple_fully_connected(A6Mp,
                                name='fc1',
                                output_dim=5,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=False,#remember regression
                                weight_decay=lambd,
                                flatten_first=True,
                                apply_relu=False)

    return Z7

def model2(X,is_training):
    '''
    DESCRIPTION:
        This model is based on the importsnce of global view
        for the prediction of variable: energy, phi eta.
        All these feature for our regression task is heavily
        dependent on the having information of whole depth at once.

        1.Energy: since the energy of particle is left in the whole
                    depth as the particle penetrates the depth of
                    detector.
        2.Phi   : Again, this signifies the orientation of the
                    momentum/trajectory direction in the x-y plane
                    and means same as phi of polar coordiate.
        3.Eta   : (pseudorapidity) is the angle that the momentum
                    vector or trajectory makes with the z axis
                    transformed by tan and log function. So,
                    ulitmately it is dependent on the whole trajectory
                    information
    USAGE:
        INPUT:
            X           : the input hits "image" to the model
            is_training : the flag to be used by batchnorm and
                            dropout for knowing whether we are in
                            training phase or testing.
        OUPUT:
            Zx          : the unnormalized and unrectified output
                            of the defined model
    '''
    #Model Hyperparameter
    bn_decision=False
    lambd=0.0
    dropout_rate=0.0

    #Reshaping the image to add the channel axis
    X_img=tf.expand_dims(X,axis=-1,name='add_channel_dim')

    #Starting the model definition
    #Adding the first layer which give the model a global view
    A1=rectified_conv3d(X_img,
                        name='conv3d1',
                        filter_shape=(3,3,40),
                        output_channel=10,
                        stride=(1,1,1),
                        padding_type='VALID',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)#it is default no need to write
    A1Mp=max_pooling3d(A1,
                        name='mpool1',
                        filter_shape=(2,2,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Adding the second layer
    A2=rectified_conv3d(A1Mp,
                        name='conv3d2',
                        filter_shape=(3,3,1),
                        output_channel=20,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A2Mp=max_pooling3d(A2,
                        name='mpool2',
                        filter_shape=(2,2,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Adding the third layer (repeating the same pattern)
    A3=rectified_conv3d(A2Mp,
                        name='conv3d3',
                        filter_shape=(3,3,1),
                        output_channel=30,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A3Mp=max_pooling3d(A3,
                        name='mpool3',
                        filter_shape=(2,2,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Adding the fourth layer (repeating the same patter as above)
    A4=rectified_conv3d(A3Mp,
                        name='conv3d4',
                        filter_shape=(3,3,1),
                        output_channel=40,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A4Mp=max_pooling3d(A4,
                        name='mpool4',
                        filter_shape=(2,2,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Adding the fifth layer (repeating the same pattern)
    A5=rectified_conv3d(A4Mp,
                        name='conv3d5',
                        filter_shape=(3,3,1),
                        output_channel=40,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A5Mp=max_pooling3d(A5,
                        name='mpool5',
                        filter_shape=(2,2,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Finally flattening and adding a fully connected layer
    A6=simple_fully_connected(A5Mp,
                                name='fc1',
                                output_dim=50,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd,
                                flatten_first=True,
                                apply_relu=True)

    #Finally the prediction/output layer
    Z7=simple_fully_connected(A6,
                                name='fc2',
                                output_dim=6,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=False,
                                weight_decay=lambd,
                                flatten_first=False, #default
                                apply_relu=False)#unnormalized

    return Z7

def model4(X,is_training):
    '''
    DESCRIPTION:
        In this model we will keep the run 15 as the baseline for the
        optimization strategy and make some changes in the model architecture.
        We will now increase the representational power of model by
        increasing the number of channels by almost double to see its effect
        of flattening of th loss curve.

        Also, we have kept the prediction of eta and phi, since it dosent
        harm the prediction of energy as seen from the comparison of run 15-16.
        So, they might also be helping each other since the regression loss
        convergence of run 15 was faster than that of run 16.
    USAGE:
        (as in the previous all models)
    '''
    #Model Hyperparameter
    bn_decision=False
    lambd=0.0
    dropout_rate=0.0

    #Reshaping the image to add the channel axis
    X_img=tf.expand_dims(X,axis=-1,name='add_channel_dim')

    #Starting the model definition
    #Adding the first layer which give the model a global view
    A1=rectified_conv3d(X_img,
                        name='conv3d1',
                        filter_shape=(3,3,40),
                        output_channel=20,  #filter size doubled
                        stride=(1,1,1),
                        padding_type='VALID',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)#it is default no need to write
    A1Mp=max_pooling3d(A1,
                        name='mpool1',
                        filter_shape=(2,2,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Adding the second layer
    A2=rectified_conv3d(A1Mp,
                        name='conv3d2',
                        filter_shape=(3,3,1),
                        output_channel=40,  #filter size doubled
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A2Mp=max_pooling3d(A2,
                        name='mpool2',
                        filter_shape=(2,2,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Adding the third layer (repeating the same pattern)
    A3=rectified_conv3d(A2Mp,
                        name='conv3d3',
                        filter_shape=(3,3,1),
                        output_channel=60,  #filter size doubled
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A3Mp=max_pooling3d(A3,
                        name='mpool3',
                        filter_shape=(2,2,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Adding the fourth layer (repeating the same patter as above)
    A4=rectified_conv3d(A3Mp,
                        name='conv3d4',
                        filter_shape=(3,3,1),
                        output_channel=80,  #filter size doubled
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A4Mp=max_pooling3d(A4,
                        name='mpool4',
                        filter_shape=(2,2,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Adding the fifth layer (repeating the same pattern)
    A5=rectified_conv3d(A4Mp,
                        name='conv3d5',
                        filter_shape=(3,3,1),
                        output_channel=80,  #filter size doubled
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A5Mp=max_pooling3d(A5,
                        name='mpool5',
                        filter_shape=(2,2,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Finally flattening and adding a fully connected layer
    A6=simple_fully_connected(A5Mp,
                                name='fc1',
                                output_dim=50,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd,
                                flatten_first=True,
                                apply_relu=True)

    #Finally the prediction/output layer
    Z7=simple_fully_connected(A6,
                                name='fc2',
                                output_dim=5,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=False,
                                weight_decay=lambd,
                                flatten_first=False, #default
                                apply_relu=False)#unnormalized

    return Z7

def model5(X,is_training):
    '''
    DESCRIPTION:
        This layer will now implement the barnd new global inception
        layer in the first layer, thus extracting the global view
        of detector hits with various receptive fields.
        Rest of the architecture will remain same as the previous
        model in this script.
    USAGE:
        INPUT:
            X           :The input to "image" of the detector
            is_training :the flag to specify whether we are in training
                            or testing mode
        OUTPUT:
            Zx          : the final un normalized activation of this
                        layer
    '''
    #Model Hyperparameter
    bn_decision=True
    lambd=0.0
    dropout_rate=0.0

    #Reshaping the image to add the channel dimension
    X_img=tf.expand_dims(X,axis=-1,name='add_channel_dim')

    #Staring the definition of model with the inception global layer
    A1=inception_global_filter_layer(X_img,
                                    name='inception_global_filter',
                                    first_filter_shape=(3,3,40),
                                    first_filter_stride=(1,1,1),
                                    second_filter_shape=(5,5,40),
                                    second_filter_stride=(1,1,1),
                                    final_channel_list=[20,10],
                                    is_training=is_training,
                                    dropout_rate=dropout_rate,
                                    apply_batchnorm=bn_decision,
                                    weight_decay=lambd)
    #Leaving the maxpooling layer for right now
    #Defining the second layer
    A2=rectified_conv3d(A1,
                        name='conv3d2',
                        filter_shape=(3,3,1),
                        output_channel=20,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A2Mp=max_pooling3d(A2,
                        name='mpool2',
                        filter_shape=(2,2,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Adding the third layer
    A3=rectified_conv3d(A2Mp,
                        name='conv3d3',
                        filter_shape=(3,3,1),
                        output_channel=30,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A3Mp=max_pooling3d(A3,
                        name='mpool3',
                        filter_shape=(2,2,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Adding the fourth layer (repeating the same patter as above)
    A4=rectified_conv3d(A3Mp,
                        name='conv3d4',
                        filter_shape=(3,3,1),
                        output_channel=40,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A4Mp=max_pooling3d(A4,
                        name='mpool4',
                        filter_shape=(2,2,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Adding the fifth layer (repeating the same pattern)
    A5=rectified_conv3d(A4Mp,
                        name='conv3d5',
                        filter_shape=(3,3,1),
                        output_channel=40,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A5Mp=max_pooling3d(A5,
                        name='mpool5',
                        filter_shape=(2,2,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Finally flattening and adding a fully connected layer
    A6=simple_fully_connected(A5Mp,
                                name='fc1',
                                output_dim=50,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd,
                                flatten_first=True,
                                apply_relu=True)

    #Finally the prediction/output layer
    Z7=simple_fully_connected(A6,
                                name='fc2',
                                output_dim=5,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=False,
                                weight_decay=lambd,
                                flatten_first=False, #default
                                apply_relu=False)#unnormalized
    return Z7


def model6(X,is_training):
    '''
    DESCRIPTION:
        This model is diretly based on the AlexNet. The only difference
        is that in the depth/layer dimension we are taking the global
        view whereas in the spatial x/y dimension we are almost following
        the Alex-Net pattern.

        This model is using urely convolution and fc layers without
        any residual or inception layers.
        Tha main difference of this model from the previous ones are that
        it is bigger in depth which might be a overclock for the current
        12k dataset.
    USAGE:
        INPUT:
            as usual
    '''
    #Model Hyperparameter
    bn_decision=False
    lambd=0.0
    dropout_rate=0.0

    #Reshaping the image to add the channel dimension
    X_img=tf.expand_dims(X,axis=-1,name='add_channel_dim')

    #Starting the first layer
    A1=rectified_conv3d(X_img,
                        name='conv3d1',
                        filter_shape=(11,11,40),
                        output_channel=96,
                        stride=(4,4,1),
                        padding_type='VALID',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)#it is default no need to write
    A1Mp=max_pooling3d(A1,
                        name='mpool1',
                        filter_shape=(3,3,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Defining the second layer
    A2=rectified_conv3d(A1Mp,
                        name='conv3d2',
                        filter_shape=(5,5,1),
                        output_channel=256,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)#it is default no need to write
    A2Mp=max_pooling3d(A2,
                        name='mpool2',
                        filter_shape=(3,3,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Defining the thord layer
    A3=rectified_conv3d(A2Mp,
                        name='conv3d3',
                        filter_shape=(3,3,1),
                        output_channel=256,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)#it is default no need to write
    A3Mp=max_pooling3d(A3,
                        name='mpool3',
                        filter_shape=(3,3,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Definign the fourth layer
    A4=rectified_conv3d(A3Mp,
                        name='conv3d4',
                        filter_shape=(3,3,1),
                        output_channel=384,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)

    #Defining the fifth layer
    A5=rectified_conv3d(A4,
                        name='conv3d5',
                        filter_shape=(3,3,1),
                        output_channel=384,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)

    #Adding the sixth layer
    A6=rectified_conv3d(A5,
                        name='conv3d6',
                        filter_shape=(3,3,1),
                        output_channel=384,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)

    #Defining the seventh layer
    A7=rectified_conv3d(A6,
                        name='conv3d7',
                        filter_shape=(3,3,1),
                        output_channel=256,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A7Mp=max_pooling3d(A7,
                        name='mpool3',
                        filter_shape=(3,3,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Now flattening and making fully connected layers
    A8=simple_fully_connected(A7Mp,
                                name='fc1',
                                output_dim=1024,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd,
                                flatten_first=True,
                                apply_relu=True)

    #Now making the socnd fully connected layer
    A9=simple_fully_connected(A8,
                                name='fc2',
                                output_dim=1024,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd,
                                apply_relu=True)

    #Now making the final layer (un-rectified version)
    Z10=simple_fully_connected(A9,
                                name='fc3',
                                output_dim=6,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd,
                                apply_relu=False)

    return Z10

def model7(X,is_training):
    '''
    DESCRIPTION:
        This function is similar to the the model6 AlexNet-global model.
        But now we want to see the effect local filter in the channel
        dimension.

        Current Hypothesis of this local Filter:
                This will be useful in the determination of the features
                of shower like curvature,torsion etc which could be useful in
                the prediciton of the barycenter output which depend on the
                direction of shower.
                Due to the lower receptive fields of the local filters they
                could be sensitive to the effect present locally which might
                get averaged out by the global filters.

                But this could have adverse effect on the prediciton of the
                energy. Let's see.
    USAGE:
        As per the previous models. as usual
    '''
    #Model Hyperparameter
    bn_decision=False
    lambd=0.0
    dropout_rate=0.0

    #Reshaping the image to add the channel dimension
    X_img=tf.expand_dims(X,axis=-1,name='add_channel_dim')

    #Now we will start defining the CNN architecture
    #Defining the input layer to the CNN
    A1=rectified_conv3d(X_img,
                        name='conv3d1',
                        filter_shape=(11,11,11),
                        output_channel=96,
                        stride=(4,4,4),
                        padding_type='VALID',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A1Mp=max_pooling3d(A1,
                        name='mpool1',
                        filter_shape=(3,3,3),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Defining the second layer
    A2=rectified_conv3d(A1Mp,
                        name='conv3d2',
                        filter_shape=(5,5,5),
                        output_channel=256,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A2Mp=max_pooling3d(A2,
                        name='mpool2',
                        filter_shape=(3,3,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Defining the third layer
    A3=rectified_conv3d(A2Mp,
                        name='conv3d3',
                        filter_shape=(3,3,3),
                        output_channel=256,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A3Mp=max_pooling3d(A3,
                        name='mpool3',
                        filter_shape=(3,3,1),
                        stride=(2,2,1),
                        padding_type='VALID')

    #Defingin the layer 4
    A4=rectified_conv3d(A3Mp,
                        name='conv3d4',
                        filter_shape=(3,3,3),
                        output_channel=384,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)

    #Applying the layer 5 without maxpooling
    A5=rectified_conv3d(A4,
                        name='conv3d5',
                        filter_shape=(3,3,3),
                        output_channel=384,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)

    #Defining the layer 6
    A6=rectified_conv3d(A5,
                        name='conv3d6',
                        filter_shape=(3,3,3),
                        output_channel=384,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)

    #Defining the layer 7 after the expansion of the channel
    A7=rectified_conv3d(A6,
                        name='conv3d7',
                        filter_shape=(3,3,3),
                        output_channel=256,
                        stride=(1,1,1),
                        padding_type='SAME',
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        apply_batchnorm=bn_decision,
                        weight_decay=lambd,
                        apply_relu=True)
    A7Mp=max_pooling3d(A7,
                        name='mpool7',
                        filter_shape=(3,3,3),
                        stride=(2,2,2),
                        padding_type='VALID')

    #Now we will flatten and put it to fully connected layers
    A8=simple_fully_connected(A7Mp,
                                name='fc1',
                                output_dim=1024,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd,
                                flatten_first=True,
                                apply_relu=True)

    #Defining the ninth layer
    A9=simple_fully_connected(A8,
                                name='fc2',
                                output_dim=1024,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd,
                                apply_relu=True)

    #Defining the last layer
    Z10=simple_fully_connected(A9,
                                name='fc3',
                                output_dim=6,
                                is_training=is_training,
                                dropout_rate=dropout_rate,
                                apply_batchnorm=bn_decision,
                                weight_decay=lambd,
                                apply_relu=False)#not normalized

    return Z10
