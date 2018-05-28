import tensorflow as  tf

################ Tensorflow Constants ###########################
#To fix a graph level for all ops to be repetable across session
graph_level_seed=1
tf.set_random_seed(graph_level_seed)
############### Global Variables ################################
#the datatype of all the variables and the placeholder
dtype=tf.float32    #this could save memory

################ Weight Initialization ##########################
def _get_variable_on_cpu(name,shape,initializer,weight_decay=None):
    '''
    DESCRIPTION:
        Since we wull be using a multi GPU, we will follow a model
        of having a shared weight among all worker GPU, which will be
        initialized in CPU.
        (See Tesnorflow website for more information about model)
        Inspiration of this function:
        https://github.com/tensorflow/models/blob/master/tutorials
                /image/cifar10/cifar10.py
    USAGE:
        INPUTS:
            name        :the name of weight variable (W/b)
            shape       :the shape of the Variable
            weight_decay:(lambda)if not None then the value specified
                            will be ysed for L2 regularization
            initializer :the name of the variable initializer
        OUTPUTS:
            weight      :the weight Variable created
    '''
    #Initializing the variable on CPU for sharing weights among GPU
    with tf.device('/cpu:0'):
        weight=tf.get_variable(name,shape=shape,dtype=dtype
                        initializer=initializer)

    if not weight_decay==None:
        #Applying the l2 regularization and multiplying with
        #the hyperpaprameter weight_decay: lambda
        reg_loss=tf.multiply(tf.nn.l2_loss(weight),weight_decay,
                                name='l2_reg_loss')
        #Adding the loss to the collection so that it could be
        #added to final loss
        tf.add_to_collection('all_losses',reg_loss)

    return weight

################ Simple Feed Forwards Layer ###################
def simple_fully_connected(X,name,output_dim,weight_decay=None,
                            is_training,apply_batchnorm=True,
                            apply_relu=True,
                        initializer=tf.glorot_uniform_initializer()):
    '''
    DESCRIPTION:
        This function will implement a simple feed-foreward network,
        taking the activation X of previous layer/input layer,
        transforming it linearly and then passing it through a
        desired non-linear activation.
    USAGE:
        INPUT:
            X           :the activation of previous layer/input layer
            output_dim  :the dimenstion of the output layer
            name        :the name of the layer
            weight_decay:(lambda) if specified to a value, then it
                            will be used for implementing the L2-
                            regularization of the weights
            is_training : to be used to state the mode i.e training or
                            inference mode.used for batch norm
            initializer :initializer choice to be used for Weights

        OUTPUT:
            A           : the activation of this layer
    '''
    with tf.name_scope(name):
        input_dim=X.get_shape().as_list()[1]
        #Checking the dimension of the input
        if not len(input_dimension)==2:
            raise AssertionError('The X should be of shape: (batch,all_nodes)')

        #Get the hold of necessary variable
        shape_W=(input_dim,output_dim)
        shape_b=(1,output_dim)
        W=_get_variable_on_cpu('W',shape_W,initializer,weight_decay)

        #Applying the linear transforamtion and passing through non-linearity
        Z=tf.multiply(X,W,'linear_transform')

        #Applying batch norm
        if apply_batchnorm==True:
            with tf.name_scope('batch_norm'):
                axis=1      #here the features are in axis 1
                Z_tilda=tf.layers.batch_normalization(Z,axis=axis,
                                                    training=is_training)
        else:
            #We generally dont regularize the bias unit
            bias_initializer=tf.zeros_initializer()
            b=_get_variable_on_cpu('b',shape_b,bias_initializer)
            Z_tilda=tf.add(Z,b,name='bias_add')

        if apply_relu==True:
            A=tf.nn.relu(Z,name='relu')
        else:
            A=Z_tilda

    return A

################ Convlutional Layers ##########################
def rectified_conv2d(X,name,filter_shape,output_channel,
                    stride,padding_type,is_training,
                    apply_batchnorm=True,weight_decay=None,apply_relu=True,
                    initializer=tf.glorot_uniform_initializer()):
    '''
    DESCRIPTION:
        This function will apply simple convolution to the given input
        images filtering the input with requires number of filters.
        This will be a custom block to apply the whole rectified
        convolutional block which include the following sequence of operation.
        conv2d --> batch_norm(optional) --> activation(optional)
    USAGE:
        INPUT:
            X              : the input 'image' to this layer. A 4D tensor of
                             shape [batch,input_height,input_width,input_channel]
            name           : the name of the this convolution layer. This will
                                be useful in grouping the components together.
                                (so currently kept as compulsory)
            filter_shape   : a tuple of form (filter_height,filter_width)
            output_channel : the total nuber of output channels in the
                             feature 'image/activation' of this layer
            stride         : a tuple giving (stride_height,stride_width)
            padding_type   : string either to do 'SAME' or 'VALID' padding
            is_training    : (used with batchnorm) a boolean to specify
                                whether we are in training or inference mode.
            apply_batchnorm: a boolean to specify whether to use batch norm or
                                not.Defaulted to True since bnorm is useful
            weight_decay   : give a value of regularization hyperpaprameter
                                i.e the amount we want to have l2-regularization
                                on the weights. defalut no regularization.
            apply_relu     : this will be useful if we dont want to apply relu
                                but some other activation function diretly
                                during the model description. Then this function
                                will not do rectification.
            initializer    : the initializer for the filter Variables
        OUTPUT:
            A       :the output feature 'image' of this layer
    '''
    with tf.name_scope(name):
        #Creating the filter weights and biases
        #Filter Weights
        input_channel=X.get_shape().as_list()[3]
        fh,fw=filter_shape
        net_filter_shape=(fh,fw,input_channel,output_channel)
        filters=_get_variable_on_cpu('W',net_filter_shape,initializer,weight_decay)

        #stride and padding configuration
        sh,sw=stride
        net_stride=(1,sh,sw,1)
        if not (padding_type=='SAME' or padding_type=='VALID'):
            raise AssertionError('Please use SAME/VALID string for padding')

        #Now applying the convolution
        Z_conv=tf.nn.conv2d(X,filters,net_stride,padding_type,name='conv')
        if apply_batchnorm==True:
            Z=_batch_normalization2d(Z_conv,is_training)
        else:
            #Biases Weight creation
            net_bias_shape=(1,1,1,output_channel)
            bias_initializer=tf.zeros_initializer()
            biases=_get_variable_on_cpu('b',net_bias_shape,bias_initializer)
            Z=tf.add(Z_conv,biases,name='bias_add')

        #Finally applying the 'relu' activation
        if apply_relu==True:
            A=tf.nn.relu(Z,name='relu')
        else:
            A=Z #when we want to apply another activation outside in model.

    return A

def max_pooling2d(X,name,filter_shape,stride,padding_type):
    '''
    DESCRIPTION:
        This function will perform maxpooling on the input 'image'
        from the previous stage of convolutional layer.The parameters
        are similar to conv2d layer.
        But there are no trainable papameters in this layer.
    USAGE:
        INPUT:

        OUTPUT:
            A       : the maxpooled map of the input 'image' with same
                        number of channels
    '''
    with tf.name_scope(name):
        #Writing the filter/kernel shape and stride in proper format
        fh,fw=filter_shape
        net_filter_shape=(1,fh,fw,1)
        sh,sw=stride
        net_stride=(1,sh,sw,1)

        #Applying maxpooling
        A=tf.nn.max_pool(A,net_filter_shape,net_stride,padding_type,name='max_pool')

    return A

def _batch_normalization2d(Z,is_training,name='batchnorm'):
    '''
    DESCRIPTION:
        (internal helper function to be used by simple conv2d)
        This function will add batch normalization on the feature map
        by normalizing the every feature map 'image' after transforming
        from the previous image to reduce the coupling between the
        two layers, thus making the current layer more roboust to the
        changes from the previous layers activation.
        (but useful with larger size,otherwise we have to seek alternative)
        like group norm etc.

        WARNING:
            we have to run a separate update op to update the rolling
            averages of moments. This has to be taken care during final
            model declaration.Else inference will not work correctly.
    USAGE:
        INPUT:
            Z           : the linear activation (convolution) of conv2d layer
            is_training : a boolean to represent whether we are in training
                            mode or inference mode.(for rolling avgs of moments)
                            (a tf.bool type usually taken as placeholder)
        OUTPUT:
            Z_tilda     : the batch-normailzed version of input
    '''
    with tf.name_scope(name):
        axis=3  #We will normalize the whole feature map across batch
        Z_tilda=tf.layers.batch_normalization(Z,axis=axis,
                                            training=is_training)
    return Z_tilda

############### Residual Layers ##############################
def identity_residual_block(X,name,num_channels,mid_filter_shape,is_training,
                            apply_batchnorm=True,weight_decay=None,
                            initializer=tf.glorot_uniform_initializer()):
    '''
    DESCRIPTION:
        This layer implements the one of the special case of residual
        layer, when the shortcut/skip connection is directly connected
        to main branch without any extra projection since dimension
        (nH,nW) dont change in the main branch.
        We will be using bottle-neck approach to reduce computational
        complexity as mentioned in the ResNet Paper.

        There are three sub-layer in this layer:
        Conv1(one-one):F1 channels ---> Conv2(fh,fw):F2 channels
                        --->Conv3(one-one):F3 channels
    USAGE:
        INPUT:
            X               : the input 'image' to this layer
            name            : the name for this identity resnet block
            channels        :the number of channels/filters in each of sub-layer
                                a tuple of (F1,F2,F3)
            mid_filter_shape: (fh,fw) a tuple of shape of the filter to be used

        OUTPUT:
            A           : the output feature map/image of this layer
    '''
    with tf.name_scope(name):
        #Applying the first one-one convolution
        A1=rectified_conv2d(X,name='branch_2a',
                            filter_shape=(1,1),
                            output_channel=num_channels[0],
                            stride=(1,1),
                            padding_type="VALID",
                            is_training=is_training,apply_batchnorm=apply_batchnorm,
                            weight_decay=weight_decay,
                            initializer=initializer)

        #Applying the Filtering in the mid sub-layer
        A2=rectified_conv2d(X,name='branch_2b',
                            filter_shape=mid_filter_shape,
                            output_channel=num_channels[1],
                            stride=(1,1),
                            padding_type="SAME",
                            is_training=is_training,apply_batchnorm=apply_batchnorm,
                            weight_decay=weight_decay,
                            initializer=initializer)

        #Again one-one convolution for upsampling
        #Sanity check for the last number of channels which should match with input
        input_channels=X.get_shape().as_list()[3]
        if not input_channels==num_channels[3]:
            raise AssertionError('Identity Block: last sub-layer channels should match input')
        Z3=rectified_conv2d(X,name='branch_2c',
                            filter_shape=(1,1),
                            output_channel=num_channels[3],
                            stride=(1,1),
                            padding_type="VALID",
                            is_training=is_training,apply_batchnorm=apply_batchnorm,
                            weight_decay=weight_decay,
                            apply_relu=False, #necessary cuz addition before activation
                            initializer=initializer)

        #Skip Connection
        #Adding the shortcut/skip connection
        with tf.name_scope('skip_conn'):
            Z=tf.add(Z3,X)
            A=tf.nn.relu(Z,name='relu')

        return A

def convolutional_residual_block(X,name,num_channels,
                            first_filter_stride,mid_filter_shape,
                            is_training,apply_batchnorm=True,weight_decay=None,
                            initializer=tf.glorot_uniform_initializer()):
    '''
    DESCRIPTION:
        This block is similar to the previous identity block but the
        only difference is that the shape (height,width) of main branch i.e 2
        is changed in the way, so we have to adust this shape in the
        skip-connection/shortcut branch also. So we will use convolution
        in the shorcut branch to match the shape.
    USAGE:
        INPUT:
            first_filter_stride : (sh,sw) stride to be used with first filter

            Rest of the argument decription is same a identity block
        OUTPUT:
            A   : the final output/feature map of this residual block
    '''
    with tf.name_scope(name):
        #Main Branch
        #Applying the first one-one convolution
        A1=rectified_conv2d(X,name='branch_2a',
                            filter_shape=(1,1),
                            output_channel=num_channels[0],
                            stride=first_filter_stride,
                            padding_type="VALID",
                            is_training=is_training,apply_batchnorm=apply_batchnorm,
                            weight_decay=weight_decay,
                            initializer=initializer)

        #Applying the Filtering in the mid sub-layer
        A2=rectified_conv2d(X,name='branch_2b',
                            filter_shape=mid_filter_shape,
                            output_channel=num_channels[1],
                            stride=(1,1),
                            padding_type="SAME",
                            is_training=is_training,apply_batchnorm=apply_batchnorm,
                            weight_decay=weight_decay,
                            initializer=initializer)

        #Again one-one convolution for upsampling
        #Here last number of channels which need not match with input
        Z3=rectified_conv2d(X,name='branch_2c',
                            filter_shape=(1,1),
                            output_channel=num_channels[3],
                            stride=(1,1),
                            padding_type="VALID",
                            is_training=is_training,apply_batchnorm=apply_batchnorm,
                            weight_decay=weight_decay,
                            apply_relu=False, #necessary cuz addition before activation
                            initializer=initializer)

        #Skip-Connection/Shortcut Branch
        #Now we have to bring the shortcut/skip-connection in shape and number of channels
        with tf.name_scope('skip_conn'):
            Z_shortcut=rectified_conv2d(X,name='branch_1',
                                filter_shape=(1,1),
                                output_channel=num_channels[3],
                                stride=first_filter_stride,
                                padding_type="VALID",
                                is_training=is_training,apply_batchnorm=apply_batchnorm,
                                weight_decay=weight_decay,
                                apply_relu=False, #necessary cuz addition before activation
                                initializer=initializer)
            #now adding the two branches element wise
            Z=tf.nn.add(Z3,Z_shortcut)
            A=tf.nn.relu(Z,name='relu')

    return A

############## Inception Module #############################
def inception_block(X,name,final_channel_list,compress_channel_list
                    is_training,apply_batchnorm=True,weight_decay=None,
                    initializer=tf.glorot_uniform_initializer()):
    '''
    DESCRIPTION:
        This block will enable us to have multiple filter's activation
        in the same layer. Multiple filters (here only 1x1,3x3,5x5 and
        a maxpooling layer) will be aplied to the input image and the
        ouptput of all these filters will be stacked in one layer.

        This is biologically inspired where we first extrct the feature
        of multiple frequencey/filter and then combine it to furthur abstract
        the idea/image.

        Filters larger than 5 not included as they will/could increase
        the computational complexity.
    USAGE:
        INPUT:
            X                   :the input image/tensor.
            name                :the name to be given this whole block.will be used in
                                    visualization
            final_channels_list : the list of channels as output of these filter
                                    [# 1x1 channels,# 3x3 channels,
                                    # 5x5 channels,# compressed maxpool channels]
            compress_channel_list: since we need to compress the input to do
                                    3x3 and 5x5 convolution. So we need the number
                                    of channels to compress into.
                                    list [#compressed channel for 3x3,
                                          #compressed channel for 5x5]
    '''
    with tf.name_scope(name):
        #Starting with the direct one-one convolution to output
        A1=rectified_conv2d(X,
                            name='1x1',
                            filter_shape=(1,1),
                            output_channel=final_channel_list[0],
                            stride=(1,1),
                            padding_type='VALID',
                            is_training=is_training,
                            apply_batchnorm=apply_batchnorm,
                            weight_decay=weight_decay,
                            apply_relu=True,
                            initializer=initializer)

        #Now starting the 3x3 convolution part
        #first compressing by 1x1
        C3=rectified_conv2d(X,
                            name='compress 3x3',
                            filter_shape=(1,1),
                            output_channel=compress_channel_list[0],
                            stride=(1,1),
                            padding_type='VALID',
                            is_training=is_training,
                            apply_batchnorm=apply_batchnorm,
                            weight_decay=weight_decay,
                            apply_relu=True,
                            initializer=initializer)
        #now doing 3x3 convolution on this compressed 'image'
        A3=rectified_conv2d(C3,
                            name='3x3',
                            filter_shape=(3,3),
                            output_channel=final_channel_list[1],
                            stride=(1,1),
                            padding_type='SAME',
                            is_training=is_training,
                            apply_batchnorm=apply_batchnorm,
                            weight_decay=weight_decay,
                            apply_relu=True,
                            initializer=initializer)

        #Now starting the same structure for the 5x5 conv part
        #first compressing by 1x1
        C5=rectified_conv2d(X,
                            name='compress 5x5',
                            filter_shape=(1,1),
                            output_channel=compress_channel_list[1],
                            stride=(1,1),
                            padding_type='VALID',
                            is_training=is_training,
                            apply_batchnorm=apply_batchnorm,
                            weight_decay=weight_decay,
                            apply_relu=True,
                            initializer=initializer)
        #now doing 5x5 convolution on this compressed 'image'
        A5=rectified_conv2d(C5,
                            name='5x5',
                            filter_shape=(5,5),
                            output_channel=final_channel_list[2],
                            stride=(1,1),
                            padding_type='SAME',
                            is_training=is_training,
                            apply_batchnorm=apply_batchnorm,
                            weight_decay=weight_decay,
                            apply_relu=True,
                            initializer=initializer)

        #Now adding the 3x3 maxpooling layer
        #first maxpooling
        CMp=max_pooling2d(X,
                          name='maxpool',
                          filter_shape=(3,3),
                          stride=(1,1),
                          padding_type='SAME')
        #now comressing to reduce channels
        AMp=rectified_conv2d(CMp,
                            name='compress maxpool',
                            filter_shape=(1,1),
                            output_channel=compress_channel_list[3],
                            stride=(1,1),
                            padding_type='VALID',
                            is_training=is_training,
                            apply_batchnorm=apply_batchnorm,
                            weight_decay=weight_decay,
                            apply_relu=True,
                            initializer=initializer)

        #Now Concatenating the sub-channels of different filter type
        concat_list=[A1,A3,A5,AMp]
        axis=-1         #across the channel axis : axis=3
        A=tf.concat(concat_list,axis=axis,name='concat')

    return A
