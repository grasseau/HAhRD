import tensorflow as tf
from tensorflow.python.client import device_lib

#import models here(need to be defined separetely in model file)
from io_pipeline import parse_tfrecords_file
from test import make_model_conv,make_model_conv3d
from test import calculate_total_loss


################## GLOBAL VARIABLES #######################
local_directory_path='datacifar/'
model_function_handle=make_model_conv

################# HELPER FUNCTIONS ########################
def _add_summary(object):
    tf.summary.histogram(object.op.name,object)

def _add_all_trainiable_var_summary():
    for var in tf.trainable_variables():
        add_summary(var)

def _get_available_gpus():
    '''
    DESCRIPTION:
        This function will extract the name of all the GPU devices
        visible to tensorflow.

        This code is inspired/copied from the following question from
        Stack Overflow:
        https://stackoverflow.com/questions/38559755/
                how-to-get-current-available-gpus-in-tensorflow
    USAGE:
        OUTPUT:
            all_gpu_name    : a list of name of all the gpu which
                                are visible to tensorflow

    '''
    local_devices=device_lib.list_local_devices()
    #Filtering the GPU devices from all the device list
    all_gpu_name=[x.name for x in local_devices
                                if x.device_type=='GPU']

    return all_gpu_name

def _get_GPU_gradient(X,Y,scope,optimizer):
    '''
    DESCRIPTION:
        This function creates a computational graph on the GPU,
        calculating the losses and the gradients on this GPUS.
    USAGE:
        INPUTS:
            X         : the input placeholder
            Y         : the target/output placeholder
            scope     : the tower scope to get the l2-reg loss
                            in its namescope
            optimizer : the optimizer function handle
    '''
    #getting the unnormalized prediction from the model
    Z=model_function_handle(X)
    #Calculating the cost of prediction form model
    total_cost=calculate_total_loss(Z,Y)

    tower_grad_var_pair=optimizer.compute_gradient(total_cost)

    return tower_grad_var_pair,total_cost

def _compute_average_gradient(all_tower_grad_var):
    '''
    DESCRIPTION:
        Following our GPU training Model. We will take the gradients
        calculated from different devices on different unique batches
        (to be verified) and get an average gradient to do
        backpropagation.
    USAGE:
        INPUT:
            all_tower_grad_var: a list of grad and var tuple for all towers
                                of form
            [
                [grad-var pair tower1],
                [grad-var pair tower2],
                ........
             ]
        OUTPUT:
            average_grad_var_pair   : the average of the gradients
                                        of each variable form all the towers
    '''
    average_grad_var_pair=[]
    #accumulating all the gradient of a var from all tower in one list by unzipping
    for all_grads_one_var in zip(*all_tower_grad_var):#unzipping
        var=all_grads_one_var[0][1] #since for same variable all the gradient are here
        all_grads=[]
        #Expanding all the gradient at axis=0 to concatenate later to get average
        for grad,_ in all_grads_one_var:
            grad=tf.expand_dims(grad,axis=0)
            all_grads.append(grad)

        #Concatenating all the gradient
        concat_all_grad=tf.concat(all_grads,axis=0)
        #Sum up the gradients to calculate the average gradient
        avg_grad=tf.reduce_mean(concat_all_grad,axis=0)

        #Creating the average grad-var pair finally.
        grad_var_pair=(avg_grad,var)
        average_grad_var_pair.append(grad_var_pair)

    return average_grad_var_pair

####################### MAIN TRAIN FUNCTION ###################
def create_training_graph(next_element):
    '''
    DESCRIPTION:
        This function will serve the main purpose of training the
        model on multiple gpu by
        1. Run inference on respective batch in each tower
        2. Calculating the loss in each tower
        3. Compute the gradient in each tower.
        4. Calculate the average gradient
        5. Calculating the moving average of all parameters
        6. Apply the gradient to minimize loss.

        This function is motivated from the CIFAR10 tutorial for
        training on multigpu on Tensorflow website.

        Please run it under the cpu:0 scope in the main driver function
    USAGE:
        INPUTS:
            next_element    : an instance of iterator.get_next() to get the next
                                batch of data from the input pipeline
        OUTPUTS:
            train_track_ops  : the list of op to run of form
                                [apply_gradient_op,loss1_op,loss2_op.....]
    '''
    #Setting the input placeholders for training mode
    #filename=tf.placeholder


    #Setting up the optimizer
    optimizer=tf.train.AdamOptimizer() #learning rate and decay Will
                                        #be added later

    #Now setting up the graph to train the model on multiple GPUs
    all_gpu_name=_get_available_gpus()  #name of all the visible GPU devices
    num_gpus=len(all_gpu_name)          #total number of GPU devices
    all_tower_grad_var=[]
    all_tower_cost=[]

    #Creating one single variable scope for all the towers
    with tf.variable_scope(tf.get_variable_scope()):
        #one by one launching the graph on each gpu devices
        for i in range(num_gpus):
            with tf.device(all_gpu_name[i]):
                with tf.name_scope('tower%s'%(i)) as tower_scope:
                    #Getting the next batch of the dataset from the iterator
                    X,Y=next_element    #'element' referes to on minibatch

                    #Create a graph on the GPU and get the gradient back
                    tower_grad_var_pair,total_cost=_get_GPU_gradient(X,Y,
                                                tower_scope,optimizer)
                    all_tower_grad_var.append(tower_grad_var_pair)
                    all_tower_cost.append(total_cost)

                    #to reuse the variable used in this tower on other tower
                    tf.get_variable_scope().reuse_variables()

    #Calculating the average gradient ffrom all the towers/devices
    #to get an average gradient to run backpropagation
    average_grad_val_pair=_compute_average_gradient(all_tower_grad_var)


    #Applying the gradient for performing backpropagation with extra dependecy
    #to update the batchnorm moving average parameter simultaneously
    extra_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        #Finally doing the backpropagation suing the optimizer
        apply_gradient_op=optimizer.apply_gradient(average_grad_val_pair)

    #Keeping the  moving average of the weight instead of the #(Hyperparameter)
    #(LATER)

    #Start checkpoint

    #Adding all the varaible summary
    _add_all_trainiable_var_summary()
    #Adding all the gradient for the summary

    #Finally accumulating all the runnable op
    train_track_ops=[apply_gradient_op]+all_tower_cost

    return train_track_ops


def train(epochs,mini_batch_size,train_filename_list,test_filename_list):
    '''
    DESCRIPTION:
        This function will finally take the graph created for training
        on multiple GPU and train that using the final training op
        and track the result using the loss of all the towers
    USAGE:
        INPUT:
            mini_batch_size     : the size of minibatch for each tower
            train_filename_list : the list of all the training tfrecords file
            test_filename_list  : the list of all the test tfrecords file
        OUTPUT:
            nothing
            later checkpoints saving will be added
    '''
    #Setting up the input_pipeline
    next_element,train_iter_init_op,test_iter_init_op=parse_tfrecords_file(
                                                        mini_batch_size,
                                                        train_filename_list,
                                                        test_filename_list)

    #Creating the multi-GPU training graph
    train_track_ops=create_training_graph(next_element)

    #initialization op for all the variable
    init=tf.global_variables_initializer()

    #Now creating the session ot run the graph
    config=tf.configProto(allow_soft_placement=True,
                          log_device_placement=True)
    with tf.Session(config) as sess:
        #initializing the global variables
        sess.run(init)

        #Starting the training epochs
        for i in range(epochs):
            #Since we are not repeating the data it will raise error once over
            while True:
                #initializing the training iterator
                sess.run(train_iter_init_op)#we need the is_training placeholder
                try:
                    track_results=sess.run(train_track_ops)
                except tf.errors.OutOfRangeError:
                    break

            #get the validation accuracy
            # while i%10==0:
            #     #starting the validation/test iterator
            #     sess.run(test_iter_init_op)
            #     try:
            #         track_results=sess.run

            #Also save the checkpoints






if __name__=='__main__':
    train_filename_list=[local_directory_path+'train.tfrecords']
    test_filename_list=[local_directory_path+'test.tfrecords']
    mini_batch_size=5

    parse_tfrecords_file(train_filename_list,test_filename_list,mini_batch_size)
