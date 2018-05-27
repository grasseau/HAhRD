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
        tf.add_to_collection('l2_reg_loss',reg_loss)

    return weight

################ Simple Feed Forwards Layer ###################
def simple_fully_connected(X,name,output_dim,weight_decay=None
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
        #We generally dont regularize the bias unit
        bias_initializer=tf.zeros_initializer()
        b=_get_variable_on_cpu('b',shape_b,bias_initializer)

        #Applying the linear transforamtion and passing through non-linearity
        Z=tf.add(tf.multiply(X,W),b,name='linear')
        A=tf.nn.relu(Z,name='relu')

    return A

################ Convlutional Layers ##########################
def simple_conv2d(X,name,filter_shape,output_channel,
                    stride,padding_type,weight_decay=None,
                    initializer=tf.glorot_uniform_initializer()):
    '''
    DESCRIPTION:
        This function will apply simple convolution to the given input
        images filtering the input with requires number of filters.
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
            weight_decay   : give a value of regularization hyperpaprameter
                                i.e the amount we want to have l2-regularization
                                on the weights. defalut no regularization.
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
        #Biases Weight creation
        net_bias_shape=(1,1,1,output_channel)
        bias_initializer=tf.zeros_initializer()
        biases=_get_variable_on_cpu('b',net_bias_shape,bias_initializer)

        #stride and padding configuration
        sh,sw=stride
        net_stride=(1,sh,sw,1)
        if not (padding_type=='SAME' or padding_type=='VALID'):
            raise AssertionError('Please use SAME/VALID string for padding')

        #Now applying the convolution
        Z_conv=tf.nn.conv2d(X,filters,net_stride,padding_type,name='conv')
        Z=tf.add(Z_conv,biases,name='bias_add')
        A=tf.nn.relu(Z,name='relu')

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

def identity_residual_block():
    '''
    DESCRIPTION:
        This layer implements the one of the special case of residual
        layer, when the shortcut/skip connection is directly connected
        to main branch without any extra projection since dimension
        (nH,nW) dont change in the main branch.
        We will be using bottle-neck approach to reduce computational
        complexity as mentioned in the ResNet Paper.
    USAGE:
        INPUT:

    '''
