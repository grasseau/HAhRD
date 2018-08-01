import tensorflow as tf
from conv2d_utils import get_variable_on_cpu

############# Tensorflow Constants #################
#this constant random seed is set for reproducability of the model
graph_level_seed=1
tf.random_seed(graph_level_seed)
############# Global COnstants #####################
dtype=tf.float32

############# RNN Layer Definition #################
def _simple_vector_RNN_cell(prev_hidden_state,
                            current_input_vector,
                            give_output,
                            output_shape,
                            name,
                            output_norm=None,
                            weight_decay=None,
                            initializer=tf.glorot_uniform_initializer()):
    '''
    DESCRIPTION:
        This function will be used by the main sequence model implementation
        block to use the RNN to give the sequential flow between the data of
        different detector layers.
    USAGE:
        INPUT:
            prev_hidden_state   : the vector from the previous RNN hidden state
            current_input_vector: the current input to the RNN (a vector)
            give_output         : a boolean specifying whether to give output
                                    from the RNN cell for not
            output_shape        : give the output shape if we want to take out
                                    the output from the current cell.
            name                : we will use this name under name scope so that
                                    we could share the variables and also group
                                    the operation in this RNN cell together
                                    for tensorboard.
            output_norm         : which norm we want to apply to the output of RNN
                                    [relu/tanh/None] supported now
            dropout_rate        : to specify the dropout rate to be applied
                                    to the new hidden state
            apply_batchnorm     : whether we want to appply batchnorm to the layer
                                    or not.
            weight_decay        : the hyperparameter to control the amount of
                                    L2 norm to be applied to the weight of the cell.
            initializer         : the initializer to initialize the weights of
                                    of the cell
        OUTPUT:
            current_hidden_state: the hidden state for the present time state
            output_vector       : (optional) the output of the RNN cell
    '''
    #We wont start a new variable scope here as we will control that from
    tf.name_scope(name):
        #Retreiving the shapes for parameter initialization
        shape_a=prev_hidden_state.get_shape().as_list()[1]
        #Retreiving the shape for current input transforamtion
        shape_x=current_input_vector.get_shape().as_list()[]

        #initializing the Waa parameters
        shape_Waa=(shape_a,shape_a)
        Waa=get_variable_on_cpu('Waa',shape_Waa,initializer,weight_decay)
        #initializing the ba parameters
        shape_ba=(1,shape_a)
        bias_initializer=tf.zeros_initializer()
        ba=get_variable_on_cpu('ba',shape_ba,bias_initializer)#we dont regularize bias unit
        #initializing the Wax parameters
        shape_Wax=(shape_x,shape_a)
        Wax=get_variable_on_cpu('Wax',shape_Wax,initializer,weight_decay)

        #Now making the transformation for next state
        Z_hidden=tf.matmul(prev_hidden_state,Waa,name='prev_mul')+\
            tf.matmul(current_input_vector,Wax,name='cur_mul')+ba

        #Now rectifying it with the tanh non-linearity
        A_hidden=tf.nn.tanh(Z_hidden)

        #Now defining the output of the current cell
        if give_output==True:
            #initializing the required parameter
            shape_Wya=(shape_a,output_shape)
            Wya=get_variable_on_cpu('Wya',shape_Wya,initializer,weight_decay)
            shape_by=(1,output_shape)
            by=get_variable_on_cpu('by',shape_by,bias_initializer)

            #Now transformaing the hidden vector to get the output
            Z_out=tf.matmul(A_hidden,Wya,name='output_mul')+by

            if output_norm=='relu':
                Z_out=tf.nn.relu(Z_out,name='relu')
            elif output_norm=='tanh':
                Z_out=tf.nn.tanh(Z_out,name='tanh')

            return A_hidden,Z_out
        else:
            return A_hidden

def _simple_vector_RNN_layer():
    '''
    DESCRIPTION:
        This is the second object in our RNN hirerchy to make the fully functional
        RNN module. This function will use the RNN cell to make a RNN layer
        and thus will be acting like an layer which takes in all the detector
        layers input making a sequence with the connection between the detector
        layer to capture the inter-layer information using the RNN cells.
    USAGE:
        INPUT:

        OUTPUT:

    '''
    

def simple_vector_RNN_block(X_img,
                            is_training,
                            conv2d_function_handle,
                            num_output_source,
                            output_dimension,
                            num_detector_layers=40,
                            initial_hidden_state='zeros'):
    '''
    DESCRIPTION:

    USAGE:
        INPUT:
            X_img                   : the 3d input image with the hits of all the
                                        detector-layers
            is_training             : whether we are in training or testing mode.
                                        used internally by batchnorm and dropout.
            conv2d_function_handle  : this will be a 2d convolutional model to be
                                        applied to the "images" of each layer of the
                                        detector with the same shared parameters
                                        and finally generating a vectored output.
            num_output_source       : whether we want an output from each detector-
                                        layer or just a single output.
                                        'all' : for number of output = number of input
                                        'one' : number of output = 1
            output_dimension        : the size of each output vector
            rnn_layers_height       : the number of RNN cells stacked on top of
                                        each other.
            num_detector_layers     : the total number of layers in detector
                                        hit-image, default to 40
            initial_hidden_state    : the initial hidden state input to be given
                                        to the first detector-layer's RNN cell.
        OUTPUT:
    '''
    #Asserting the dimension of input with the number of detector-layer
    layer_dim=3
    assert len(X_img.get_shape().as_list()),'give in: [batch,x,y,z] format'
    assert X_img.get_shape().as_list()[layer_dim]==num_detector_layers,'detector\
                                        layer number mismatch'

    #Running the convolution on the same varaible scope for each detector layer
    conv_output_list=[]
    with tf.variable_scope('shared_conv_layers'):
        for i in range(num_detector_layers):
            #Running the convolution on each layers of the detector one by one
            #Slicing the input image for getting a detector layer
            X=X_img[:,:,:,i]
            #Passing it through the convolution layer with shared parameters
            det_layer_activation=conv2d_function_handle(X,is_training)

            #Appending the output of current layers convolution to the list
            conv_output_list.append(det_layer_activation)

            #reusing the variable/parameters of CNN for other detecotr layers
            tf.get_variable_scope().reuse_variables()

    #Now we are ready for the implementation of the sequence(RNN/LSTM) cells
