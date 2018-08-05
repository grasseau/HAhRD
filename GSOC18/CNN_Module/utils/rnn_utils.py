import tensorflow as tf
from conv2d_utils import get_variable_on_cpu

################### Tensorflow Constants #################
#this constant random seed is set for reproducability of the model
graph_level_seed=1
tf.set_random_seed(graph_level_seed)
################### Global COnstants #####################
dtype=tf.float32

################### RNN Layer Definition #################
def _simple_vector_RNN_cell(prev_hidden_state,
                            current_input_vector,
                            give_output,
                            output_shape,
                            name,
                            output_norm,
                            weight_decay,
                            initializer):
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
            weight_decay        : the hyperparameter to control the amount of
                                    L2 norm to be applied to the weight of the cell.
            initializer         : the initializer to initialize the weights of
                                    of the cell
        OUTPUT:
            current_hidden_state: the hidden state for the present time state
            output_vector       : (optional) the output of the RNN cell
    '''
    #We wont start a new variable scope here as we will control that from
    with tf.name_scope(name):
        #Retreiving the shapes for parameter initialization
        shape_a=prev_hidden_state.get_shape().as_list()[1]
        #Retreiving the shape for current input transforamtion
        shape_x=current_input_vector.get_shape().as_list()[1]

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
        #initializing the Wya and by parameter
        shape_Wya=(shape_a,output_shape)
        Wya=get_variable_on_cpu('Wya',shape_Wya,initializer,weight_decay)
        #getting by parameters
        shape_by=(1,output_shape)
        by=get_variable_on_cpu('by',shape_by,bias_initializer)

        #Now making the transformation for next state
        Z_hidden=tf.matmul(prev_hidden_state,Waa,name='prev_mul')+\
            tf.matmul(current_input_vector,Wax,name='cur_mul')+ba

        #Now rectifying it with the tanh non-linearity
        A_hidden=tf.nn.tanh(Z_hidden)

        #Now defining the output of the current cell
        if give_output==True:
            #Now transformaing the hidden vector to get the output
            Z_out=tf.matmul(A_hidden,Wya,name='output_mul')+by

            if output_norm=='relu':
                Z_out=tf.nn.relu(Z_out,name='relu')
            elif output_norm=='tanh':
                Z_out=tf.nn.tanh(Z_out,name='tanh')

            return A_hidden,Z_out
        else:
            return A_hidden

def _simple_vector_RNN_layer(input_sequence,
                                name,
                                hidden_state_length,
                                num_output_source,
                                output_dimension,
                                output_norm,
                                weight_decay,
                                initializer):
    '''
    DESCRIPTION:
        This is the second object in our RNN hirerchy to make the fully functional
        RNN module. This function will use the RNN cell to make a RNN layer
        and thus will be acting like an layer which takes in all the detector
        layers input making a sequence with the connection between the detector
        layer to capture the inter-layer information using the RNN cells.
    USAGE:
        INPUT:
            input_sequence      : the input sequence to this RNN "layer".This
                                    will include the activation of all the detector-
                                    layer in general case.
            name                : this name will be used to create a shared variable
                                    scope for all the RNN cells inside this layer
            hidden_state_length : the length of the hidden state vector which will
                                    carry over the memory between layers.
            num_output_source   : whether we want an output from each detector-
                                    layer or just a single output.
                                    'all' : for number of output = number of input
                                    'one' : number of output = 1
            output_dimension    : the size of each output vector
            output_norm         : the normalization to be applied to the output
                                    of the RNN cell.
                                    Currently relu and tanh supported.
            weight_decay        : the factor by which the L2-regularization of the
                                    weights will be multipled.
            initializer         : the initializer function handle for the weights/
                                    parameter in the RNN cell. default to
                                    glorot uniform initializer.
        OUTPUT:

    '''
    #Now starting the shared variable scope for all the RNN cells
    with tf.variable_scope(name):
        #creating the initial hidden state which will be a zero vector
        hidden_state_shape=(1,hidden_state_length)
        prev_hidden_state=tf.constant(0.0,shape=hidden_state_shape,dtype=dtype)

        #Now will will initiate the RNN connncetion to get inter-detector_layer communication
        #Making a list to hold the output sequences.
        output_sequence=[]

        #Setting up the give output parameter
        assert num_output_source=='all' or num_output_source=='one','Give correct\
                                                            arguments'
        give_output=False
        if num_output_source=='all':
            give_output=True

        #Iterating over the all the sequences to have communication between them
        for seq_i in range(len(input_sequence)):
            #Initializing the RNN cell under the above variable scope
            if seq_i==len(input_sequence)-1:
                #Atleast the last RNN cell has to give output
                give_output=True

            output_tuple=_simple_vector_RNN_cell(prev_hidden_state,
                                                input_sequence[seq_i],
                                                give_output=give_output,
                                                output_shape=output_dimension,
                                                name='RNN{}'.format(seq_i+1),
                                                output_norm=output_norm,
                                                weight_decay=weight_decay,
                                                initializer=initializer)
            if give_output==True:
                prev_hidden_state=output_tuple[0]
                output_sequence.append(output_tuple[1])
            else:
                prev_hidden_state=output_tuple

            #Sharing the parameter in the current varaible scope across RNN cells
            tf.get_variable_scope().reuse_variables()

        #The output sequence from this RNN "layer" is ready to given out
        return output_sequence

################### LSTM Layer Definition ##################
def _simple_vector_LSTM_cell(prev_memory_state,
                            prev_hidden_state,
                            current_input_vector,
                            give_output,
                            output_shape,
                            name,
                            output_norm,
                            weight_decay,
                            initializer):
    '''
    DESCRIPTION:
        This function will describe an ideal LSTM cell which is
        analogous to our previous development heirarchy
        cell->layer->block.

        The input and output format of a cell be customized according
        to the typical LSTM block with two ingoing connection
        1.prev_memory_state
        2.prev_hidden_state
        and it will output the corresponding two state as output
        with an optional output from the cell to next layer.
        (if we are making this cell to do sequence-to-sequence connection)
    USAGE:
        INPUT:
            prev_memory_state   : this will be the memory state of the previous
                                    LSTM cell.
            prev_hidden_state   : this will be the hidden state of the previous
                                    LSTM cell.
            current_input_vector: this will be the input to the LSTM cell at the
                                    current time.
            give_output         : a boolean to specify whether we want to give
                                    output from the current LSTM cell.
            output_shape        : (if the current LSTM cell is giving output)then
                                    what is the shape/size of output vector.
            name                : the name to be given to the current LSTM cell
                                    for simpler visualization in the tensorboard.
                                    This will just open a name scope since we will
                                    be sharing the weight among all the LSTM cells
                                    in the layer.
            output_norm         : the normalization to be applied to the (optional)
                                    output of this layer.
            weight_decay        : the scalar to be multiplied to the L2-regularization
                                    cost to control the weight decay.
            initializer         : the initiailizer to be used to initialize the
                                    parameters/weights of this layer.
    '''
    #Starting the cell in name scope to encapsulate the cell visualization
    with tf.name_scope(name):
        #Extracting the shapes to be used for making weights/parameter
        na=prev_memory_state.get_shape().as_list()[1]
        nx=current_input_vector.get_shape().as_list()[1]
        #Initializing the handle for bias initializer
        bias_initializer=tf.zeros_initializer()


        #Making/Retreiving the required parameters
        #Candidate Parameters
        shape_Wc=(na+nx,na)
        shape_bc=(1,na)
        Wc=get_variable_on_cpu('Wc',shape_Wc,initializer,weight_decay)
        bc=get_variable_on_cpu('bc',shape_bc,bias_initializer)
        #Update Gate Parameters
        shape_Wu=(na+nx,na)
        shape_bu=(1,na)
        Wu=get_variable_on_cpu('Wu',shape_Wu,initializer,weight_decay)
        bu=get_variable_on_cpu('bu',shape_bu,bias_initializer)
        #Forget Gate Parameter
        shape_Wf=(na+nx,na)
        shape_bf=(1,na)
        Wf=get_variable_on_cpu('Wf',shape_Wf,initializer,weight_decay)
        bf=get_variable_on_cpu('bf',shape_bf,bias_initializer)
        #Memory-Output Gate Parameters
        shape_Wo=(na+nx,na)
        shape_bo=(1,na)
        Wo=get_variable_on_cpu('Wo',shape_Wo,initializer,weight_decay)
        bo=get_variable_on_cpu('bo',shape_bo,bias_initializer)
        #Cell-Output Parameters
        shape_Wy=(na,output_shape)
        shape_by=(1,output_shape)
        Wy=get_variable_on_cpu('Wy',shape_Wy,initializer,weight_decay)
        by=get_variable_on_cpu('by',shape_by,bias_initializer)

        #Now performig the necessary gate operation
        #Concatenating the prev_hidden_state and input along axis=1
        IH_concat_vec=tf.concat([prev_hidden_state,current_input_vector],
                                axis=1,
                                name='Inp/Hidd_concat')

        #Calculating the next memory candidate
        with tf.name_scope('new_memory'):
            with tf.name_scope('candidate_calc'):
                Zc_delta=tf.matmul(IH_concat_vec,Wc)+bc
                c_delta=tf.nn.tanh(Zc_delta)
            #Calculation of the update gate
            with tf.name_scope('update_gate'):
                Z_delta_u=tf.matmul(IH_concat_vec,Wu)+bu
                delta_u=tf.nn.sigmoid(Z_delta_u)
            #Calculation of the forget gate
            with tf.name_scope('forget_gate'):
                Z_delta_f=tf.matmul(IH_concat_vec,Wf)+bf
                delta_f=tf.nn.sigmoid(Z_delta_f)
            #Finally calculation of the candidate
            with tf.name_scope('memory_update'):
                c_t=delta_f*prev_memory_state+delta_u*c_delta

        #Now calculation of the new hidden state
        with tf.name_scope('new_hidden_state'):
            with tf.name_scope('output_gate'):
                Z_delta_o=tf.matmul(IH_concat_vec,Wo)+bo
                delta_o=tf.nn.sigmoid(Z_delta_o)
            with tf.name_scope('hidden_state_update'):
                a_t=tf.nn.tanh(c_t)*delta_o

        #Now finally its time for optional output
        if give_output==True:
            with tf.name_scope('cell_output'):
                Z_yt=tf.matmul(a_t,Wy)+by
                if output_norm=='relu':
                    Z_yt=tf.nn.relu(Z_yt)
                elif output_norm=='tanh':
                    Z_yt=tf.nn.tanh(Z_yt)

            return [c_t,a_t,Z_yt]
        else:
            return [c_t,a_t]

def _simple_vector_LSTM_layer(input_sequence,
                                name,
                                hidden_state_length,
                                num_output_source,
                                output_dimension,
                                output_norm,
                                weight_decay,
                                initializer):
    '''
    DESCRIPTION:
        This function will be the second building block in the hierarchy
        of the LSTM Module. This will encapsulate the whole layer
        of the LSTM cells as we did for the RNN cells.
        The enetry and exit points for this layer will be:
        1.input_sequence
        2.output_sequence (or just a single output)

        Also, we will taking the length of the memory vector to be
        equal to the hidden state vector of LSTM cell, which is a common
        desgn pattern.

    USAGE:
        INPUT:
            input_sequence      : this will give the LSTM layer the sequence of
                                    of inputs from the previous layer or CNN layer
            name                : this name will be used to invoke a unique variable
                                    scope for sharing the weights among the LSTM cells
            hidden_state_length : = memory_vector_length
            num_output_source   : whether we want 'all' the cells to give output
                                    or just 'one' last cell of the layer.
                                    ['all' / 'one']
            output_dimension    : the length of the output vector form the cells
            output_norm         : which of the normalizaiton ot be applied to
                                    all the output of the layer's cell
            weight_decay        : the weight decay parameter to control the
                                    L2-regularization of the weights
            initializer         : the initializer to be used by the LSTM cells
                                    to initialize their parameters/weights
        OUTPUT:
            output_sequence     : the sequence(or just one) of output of the
                                    current cell.
    '''
    #Starting the variable scope to share the parameters among the LSTM cells
    with tf.variable_scope(name):
        #Initializing the initial memory and hidden states of the layer
        batch_size=input_sequence[0].get_shape().as_list()[0]
        state_shape=(batch_size,hidden_state_length)
        prev_memory_state=tf.constant(0.0,shape=state_shape,dtype=dtype)
        prev_hidden_state=tf.constant(0.0,shape=state_shape,dtype=dtype)

        #Initilaizing the list to collect the output_sequence
        output_sequence=[]

        #Asserting the corect input arguments
        assert num_output_source=='all' or num_output_source=='one',\
                        'Please give the correct number of output source'

        #setting the give_output parameters for all the cells of layer
        give_output=False
        if num_output_source=='all':
            give_output=True

        #Starting to create the LSTM sequence
        for seq_i in range(len(input_sequence)):
            #Setting up the give_output for the compulsary output
            if seq_i==len(input_sequence)-1:
                give_output=True

            cell_output=_simple_vector_LSTM_cell(prev_memory_state,
                                            prev_hidden_state,
                                            input_sequence[seq_i],
                                            give_output=give_output,
                                            output_shape=output_dimension,
                                            name='LSTM{}'.format(seq_i),
                                            output_norm=output_norm,
                                            weight_decay=weight_decay,
                                            initializer=initializer)

            #Now setting up the memory and hidden state for next cell in seq
            prev_memory_state=cell_output[0]
            prev_hidden_state=cell_output[1]
            #Now appending the output of the cell if possible
            if give_output==True:
                output_sequence.append(cell_output[2])

            #Specifying the layer to reuse the parameter in each cell
            tf.get_variable_scope().reuse_variables()

        #Finally returning the sequence for the next layer or output
        return output_sequence


################# RNN BLOCK Definition ####################
def _tfwhile_cond(X_img,is_training,iter_i,iter_end,reg_loss,tensor_array):
    '''
    DESCRIPTION:
        This callable will be used in the simple_vector_RNN_block to make the
        CNN graph using the tf.while_loop. Though this tf.while loop makes the
        unrolled representation of the computation of the graph it gives two
        powerfull control to us to control the speed and memory
        1.parallel_iteration :for parallely executing the branches of the loop
        2.swap_memory        : for swapping the foreqard propagation tensor
                                from CPU to GPU for time being before being used
                                in the backpropagation.
    USAGE:
        Dont use it directly. it will be called with the arguments of the
        tf.while_loop loop_vars.
    '''
    return tf.less(iter_i,iter_end)

def _tfwhile_body(X_img,is_training,iter_i,iter_end,tensor_array):
    '''
    DESCRIPTION:
        Again this function will be called by the tf.while loop to perform the
        2D convolution on the layers of the detector and keep concatenating the
        output vector-encoding of the CNN operation on each layer which will
        be later used by the RNN Module.

        (maybe we have to use the parallel_iteration=1 to not cause the randomization
        in the concatenation. But since all the current layer's encoding concat
        depends on the all the previous concat ebing done, then the parallel_iteration
        should not cause the problem)
    USAGE:
        Don't use it directly it will be called by the tf.while
    '''

    #Running the convolution on each layers of the detector one by one
    #Slicing the input image for getting a detector layer
    X=tf.expand_dims(X_img[:,:,:,iter_i],axis=-1,name='channel_dim')

    #Passing it through the convolution layer with shared parameters
    det_layer_activation=conv2d_function_handle(X,is_training)
    #Expanding the output activation to be ready for the result concatenation
    det_layer_activation=tf.expand_dims(det_layer_activation,axis=2)

    #adding the output encoding to the tensorarray
    tensor_array=tensor_array.write(iter_i,det_layer_activation)

    #Updating the iter_i
    iter_i=iter_i+1

    #returning all the loop_vars
    return [X_img,iter_i,iter_end,tensor_array]


    #The shared variable scope is being started outside the tf.while loop
    # #reusing the variable/parameters of CNN for other detecotr layers
    # tf.get_variable_scope().reuse_variables()



def simple_vector_RNN_block(X_img,
                            is_training,
                            conv2d_function_handle,
                            sequence_model_type,
                            num_of_sequence_layers,
                            hidden_state_dim_list,
                            output_dimension_list,
                            output_type,
                            output_norm_list,
                            num_detector_layers=40,
                            weight_decay=None,
                            initializer=tf.glorot_uniform_initializer()):
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
            sequence_model_type     : whether we want to use the 'LSTM' layers
                                        or RNN layers
                                        (later support for mixing both could be
                                        added).
                                        ['LSTM'/'RNN']
            num_of_sequence_layers  : the number of RNN cells stacked on top of
                                        each other.
                                        Practically <=2 is best to keep.
            hidden_state_dim_list   : the list containing the dimension of the
                                        hidden state of the RNN layers.
            output_dimension_list   : the list containing the output dimension of
                                        each RNN layer
            output_type             : whether we want to return a sequence or
                                        vector as output of this block.
                                        string: 'sequence'/'vector'
            output_norm_list        : the name of the normalization to be applied
                                        to the output of each layer.
                                        ['relu'/'tanh'/None] supported now
            num_detector_layers     : the total number of layers in detector
                                        hit-image, default to 40
            weight_decay            : the hyperparameter that will be multiplied
                                        to the L2-regularization contribution
                                        to the total cost.
            initializer              : the initializer to initialize the weigths
                                        of the weights in the layers (CNN and RNN)
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
        #initializing the iter varaible
        iter_i=tf.constant(0,dtype=tf.int32,name='iter_i')
        iter_end=tf.constant(num_detector_layers,dtype=tf.int32,name='iter_end')

        #Initializing the TensorArray for holding all the layer activation
        tensor_array=tf.TensorArray(dtype=dtype,
                                    size=num_detector_layers,
                                    clear_after_read=True,#no read many
                                    infer_shape=True)

        #Initializing the constant to hold the regularization loss
        conv_reg_loss=tf.constant(0.0,dtype=dtype,name='reg_loss_value')

        #Now running the tf.while loop and the final tensor array as the output
        _,_,_,_,conv_reg_loss,tensor_array=tf.while_loop(_tfwhile_cond,
                                        #_tfwhile_body,
                                        conv2d_function_handle,
                                        loop_vars=[X_img,is_training,iter_i,iter_end,\
                                                conv_reg_loss,tensor_array],
                                        #none of them will be shape invaraint,
                                        swap_memory=True,
                                        parallel_iterations=16)

        #Now adding this regularization of this conv_layer to one collection
        tf.add_to_collection('reg_losses',conv_reg_loss)
        tf.summary.scalar('l2_reg_loss_conv',conv_reg_loss)


    #Now we are ready for the implementation of the sequence(RNN/LSTM) cells
    #Retreiving the sequence tensor (vector-encoding) from the tensor array
    cnn_output_vectors=tensor_array.stack()
    input_sequence=[cnn_output_vectors[i,:,:] for i in range(num_detector_layers)]

    #All the necessary argument assertion
    assert output_type=='sequence' or output_type=='vector','Give correct argument'

    #Writing in a separate name scope since varaible scope are taken care inside
    with tf.name_scope('seq_RNN_layers') as rnn_block_scope:
        #Stacking up the RNN seq-layer on top on one another.
        for i in range(num_of_sequence_layers):
            #Deciding the unique layer name for unique varaible scope for each layer
            layer_name='layer{}'.format(i+1)

            #Specifying the number of output_source
            num_output_source='all'
            if i==num_of_sequence_layers-1 and output_type=='vector':
                num_output_source='one'

            #Now stacking up the layers on top of other
            #Selecting the type of sequence model we want stack
            seq_layer_function_handle=None
            if sequence_model_type=='RNN':
                seq_layer_function_handle=_simple_vector_RNN_layer
            elif sequence_model_type=='LSTM':
                seq_layer_function_handle=_simple_vector_LSTM_layer

            #The output of this layer will be the input sequence to next RNN layer
            input_sequence=seq_layer_function_handle(input_sequence=input_sequence,
                                    name=layer_name,
                                    hidden_state_length=hidden_state_dim_list[i],
                                    num_output_source=num_output_source,
                                    output_dimension=output_dimension_list[i],
                                    output_norm=output_norm_list[i],
                                    weight_decay=weight_decay,
                                    initializer=initializer)

        #Finally returning the output sequence be it a list of one vector or all
        output_sequence=input_sequence

        #Adding the regularization loss of this scope to the
        reg_loss_list_rnn=tf.get_collection('all_losses',scope=rnn_block_scope)
        l2_reg_loss_rnn=0.0
        if not len(reg_loss_list_rnn)==0:
            l2_reg_loss_rnn=tf.add_n(reg_loss_list_rnn,name='l2_reg_loss_rnn')
        #Adding this regularization loss to the reg_losses collection
        tf.add_to_collection('reg_losses',l2_reg_loss_rnn)
        tf.summary.scalar('l2_reg_loss_rnn',l2_reg_loss_rnn)


        #This output could be used for furthur fully connected layer/
        #aggregateion (if its a sequence) or input to other sequence layer
        #or directly as the unnormalized output of the whole model.
        return output_sequence
