import tensorflow as tf
from tensorflow.python.client import device_lib

#import models here(need to be defined separetely in model file)
from test import make_model_conv,make_model_conv3d
from test import calculate_total_loss

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

def _get_GPU_gradient(X,Y,optimizer):
    '''
    DESCRIPTION:
        This function creates a computational graph on the GPU,
        calculating the losses and the gradients on this GPUS.
    USAGE:
        INPUTS:
            X         : the input placeholder
            Y         : the target/output placeholder
            optimizer : the optimizer function handle
    '''
    #getting the unnormalized prediction from the model
    Z=make_model_conv(X)
    #Calculating the cost of prediction form model
    total_cost=calculate_total_loss(Z,Y)

    tower_grad_var_pair=optimizer.compute_gradient(total_cost)

    return tower_grad_var_pair

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

def train(X,Y):
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
            X       : the input placeholder to the model
            Y       : the target lable placeholder of model

    '''
    #Setting up the optimizer
    optimizer=tf.train.AdamOptimizer() #learning rate and decay Will
                                        #be added later

    #Now setting up the graph to train the model on multiple GPUs
    all_gpu_name=_get_available_gpus()  #name of all the visible GPU devices
    num_gpus=len(all_gpu_name)          #total number of GPU devices
    all_tower_grad_var=[]

    #Creating one single variable scope for all the towers
    with tf.variable_scope(tf.get_variable_scope()):
        #one by one launching the graph on each gpu devices
        for i in range(num_gpus):
            with tf.device(all_gpu_name[i]):
                with tf.name_scope('tower%s'%(i)):
                    #Create a graph on the GPU and get the gradient back
                    tower_grad_var_pair=_get_GPU_gradient(X,Y,optimizer)
                    all_tower_grad_var.append(tower_grad_var_pair)
                    #to reuse the variable used in this tower on other tower
                    tf.get_variable_scope().reuse_variables()

    #Calculating the average gradient ffrom all the towers/devices
    #to get an average gradient to run backpropagation
    average_grad_val_pair=_compute_average_gradient(all_tower_grad_var)

    #Applying the gradient for performing backpropagation
    apply_gradient_op=optimizer.apply_gradient(average_grad_val_pair)

    #Keeping the  moving average of the weight instead of the
    #actual weight to remove any noisy update which may come from
    #batch backpropagation
    #(Hyperparameter)
