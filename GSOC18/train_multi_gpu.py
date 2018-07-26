import tensorflow as tf
import datetime
import sys
import os
from tensorflow.python.client import device_lib
from tensorflow.python.client import timeline

#import models here(need to be defined separetely in model file)
from CNN_module/utils/io_pipeline import parse_tfrecords_file
# from test import make_model_conv,make_model_conv3d,make_model_linear
# from test import calculate_model_accuracy,calculate_total_loss
# from model1_definition import model7 as model_function_handle
# from model1_definition import calculate_model_accuracy
# from model1_definition import calculate_total_loss


################## GLOBAL VARIABLES #######################
# local_directory_path='/home/gridcl/kumar/HAhRD/GSOC18/GeometryUtilities-master/interpolation/image_data'
# run_number=35                            #for saving the summaries
# train_summary_filename='tmp/hgcal/%s/train/'%(run_number) #for training set
# test_summary_filename='tmp/hgcal/%s/valid/'%(run_number)  #For validation set
# if os.path.exists(train_summary_filename):
#     os.system('rm -rf ./'+train_summary_filename)
#     os.system('rm -rf ./'+test_summary_filename)
#
# checkpoint_filename='tmp/hgcal/{}/checkpoint/'.format(run_number)
# #model_function_handle=model2
# timeline_filename='tmp/hgcal/{}/timeline/'.format(run_number)
# if not os.path.exists(timeline_filename):
#     os.makedirs(timeline_filename)

################# HELPER FUNCTIONS ########################
def _add_summary(object):
    tf.summary.histogram(object.op.name,object)

def _add_all_trainiable_var_summary():
    for var in tf.trainable_variables():
        _add_summary(var)

def _add_all_gpu_losses_summary(train_track_ops):
    for loss in train_track_ops[1:]:
        _add_summary(loss)

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

def _get_GPU_gradient(model_function_handle,
                        calculate_model_accuracy,
                        calculate_total_loss,
                        X,Y,is_training,scope,optimizer):
    '''
    DESCRIPTION:
        This function creates a computational graph on the GPU,
        calculating the losses and the gradients on this GPUS.
    USAGE:
        INPUTS:
            X         : the input placeholder
            Y         : the target/output placeholder
            is_training: the
            scope     : the tower scope to get the l2-reg loss
                            in its namescope
            optimizer : the optimizer function handle
    '''
    #getting the unnormalized prediction from the model
    Z=model_function_handle(X,is_training)

    #Calculating the accuracy of the model
    accuracy=calculate_model_accuracy(Z,Y)

    #Calculating the cost of prediction form model
    total_cost=calculate_total_loss(Z,Y,scope)

    tower_grad_var_pair=optimizer.compute_gradients(total_cost)

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
def create_training_graph(model_function_handle,
                            calculate_model_accuracy,
                            calculate_total_loss,
                            iterator,is_training,
                            global_step,learning_rate):
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
            model_function_handle   : the fucntion handle to create CNN graph
            calculate_model_accuracy: the fucntion handle to calculate the model
                                        accuracy
            calculate_total_loss    : the fucntion handle to calculate the total
                                        loaa of the CNN  model
            iterator                : an instance of iterator having the property of
                                        .get_next() to get the next
                                        batch of data from the input pipeline
                                        (following Derek Murray advice on Stack Overflow)
            is_training             : a boolean flag to distinguish in what mode we are in
                                        Training/Testing
            global_step             : a varible which could how many rounds of backpropagation
                                        is completed
            learning_rate           : the learning rate with the learning rate decay
                                        applied to it
        OUTPUTS:
            train_track_ops         : the list of op to run of form
                                        [apply_gradient_op,loss1_op,loss2_op.....]
    '''
    #Setting up the optimizer
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)

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
                    X,Y=iterator.get_next() #'element' referes to on minibatch

                    #Create a graph on the GPU and get the gradient back
                    tower_grad_var_pair,total_cost=_get_GPU_gradient(
                                            model_function_handle,
                                            calculate_model_accuracy,
                                            calculate_total_loss,
                                            X,Y,
                                            is_training,tower_scope,optimizer)
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
        apply_gradient_op=optimizer.apply_gradients(average_grad_val_pair,
                                                global_step=global_step)

    #Keeping the  moving average of the weight instead of the #(Hyperparameter)
    #(LATER)

    #Adding all the gradient for the summary

    #Finally accumulating all the runnable op
    train_track_ops=[apply_gradient_op]+all_tower_cost

    return train_track_ops


def train(run_number,
            model_function_handle,
            calculate_model_accuracy,
            calculate_total_loss,
            epochs,mini_batch_size,shuffle_buffer_size,
            init_learning_rate,decay_step,decay_rate,
            train_filename_list,test_filename_list,
            restore_epoch_number=None):
    '''
    DESCRIPTION:
        This function will finally take the graph created for training
        on multiple GPU and train that using the final training op
        and track the result using the loss of all the towers
    USAGE:
        INPUT:
            run_number                : the run number to save the results and
                                        checkpoints in unique directory
            model_function_handle     : the fucntion handle to create the CNN
                                        model architecture
            calculate_model_accuracy  : the fucntion handle to calculate the
                                        accuracy of the model
            calculate_total_loss      : the function handle to calculate the
                                        loss or cost of the model
            epochs                    : the total number of epochs to go through
                                        the dataset
            mini_batch_size           : the size of minibatch for each tower
            shuffle_buffer_size       : the buffer size to randomly sample minibatch
            init_learning_rate        : the initial learning rate of the oprimizer
            decay_step                : the number of step after which one unit decay
                                        will be applied to learning_rate
            decay_rate                : the rate at which the the learning rate
                                        will be scaled acc to inbuilt formula
            train_image_filename_list : list of all the training image tfrecords
                                        file
            train_label_filename_list : list of the filename of label of the
                                        training image
            test_image_filename_list  : the list of all the test images
                                        tfrecords file
            test_label_filename_list  : the list of the filename having
                                        corresponding labels for the images
            restore_epoch_number      : the number if given will be used for
                                        restoring the training.
        OUTPUT:
            nothing
            later checkpoints saving will be added
    '''
    #Setting up the required directory for savinng results and checkpoint
    train_summary_filename='tmp/hgcal/%s/train/'%(run_number) #for training set
    test_summary_filename='tmp/hgcal/%s/valid/'%(run_number)  #For validation set
    if os.path.exists(train_summary_filename):
        os.system('rm -rf ./'+train_summary_filename)
        os.system('rm -rf ./'+test_summary_filename)

    checkpoint_filename='tmp/hgcal/{}/checkpoint/'.format(run_number)
    #model_function_handle=model2
    timeline_filename='tmp/hgcal/{}/timeline/'.format(run_number)
    if not os.path.exists(timeline_filename):
        os.makedirs(timeline_filename)

    #Setting the required Placeholders and Global Non-Trainable Variable
    is_training=tf.placeholder(tf.bool,[],name='training_flag')
    global_step=tf.get_variable('global_step',shape=[],
                        initializer=tf.constant_initializer(0),
                        trainable=False)
    learning_rate=tf.train.exponential_decay(init_learning_rate,
                                            global_step,
                                            decay_step,
                                            decay_rate,#lr_decay rate
                                            staircase=True,
                                            name='exponential_decay')
    tf.summary.scalar('learning_rate',learning_rate)

    #Setting up the input_pipeline
    with tf.name_scope('IO_Pipeline'):
        iterator,train_iter_init_op,test_iter_init_op=parse_tfrecords_file(
                                                    train_filename_list,
                                                    test_filename_list,
                                                    mini_batch_size,
                                                    shuffle_buffer_size=shuffle_buffer_size)

    #Creating the multi-GPU training graph
    train_track_ops=create_training_graph(model_function_handle,
                                            calculate_model_accuracy,
                                            calculate_total_loss,
                                            iterator,is_training,global_step,
                                            learning_rate)

    #Adding saver to create checkpoints for weights
    saver=tf.train.Saver(tf.global_variables(),
                        max_to_keep=2)

    #Adding all the varaible summary
    train_writer=tf.summary.FileWriter(train_summary_filename)
    test_writer=tf.summary.FileWriter(test_summary_filename)
    _add_all_trainiable_var_summary()
    _add_all_gpu_losses_summary(train_track_ops)
    merged_summary=tf.summary.merge_all()
    #Adding the merged summary in train-track ops
    train_track_ops.append(merged_summary)

    #initialization op for all the variable
    init=tf.global_variables_initializer()

    #Now creating the session ot run the graph
    config=tf.ConfigProto(allow_soft_placement=True,
                          log_device_placement=False)
    with tf.Session(config=config) as sess:
        if restore_epoch_number==None:
            #initializing the global variables
            sess.run(init)
        else:
            #Restoring the saved model if possible
            checkpoint_path=checkpoint_filename+'model.ckpt-%s'%(restore_epoch_number)
            saver.restore(sess,checkpoint_path)

        #Adding the graph to tensorborad
        train_writer.add_graph(sess.graph)
        test_writer.add_graph(sess.graph)

        #Starting the training epochs
        for i in range(epochs):
            #Since we are not repeating the data it will raise error once over
            #initializing the training iterator
            sess.run(train_iter_init_op) #we need the is_training placeholder
            bno=1                        #writing the batch number
            t_epoch_start=datetime.datetime.now()
            while True:
                try:
                    #Running the train op and optionally the tracer bullet
                    if bno%20==0 and i%5==0:
                        #Adding the runtime statisctics (memory and execution time)
                        run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata=tf.RunMetadata()

                        #Starting the timer
                        t0=datetime.datetime.now()
                        #Running the op
                        track_results=sess.run(train_track_ops,
                                                feed_dict={is_training:True},
                                                options=run_options,
                                                run_metadata=run_metadata)
                        t1=datetime.datetime.now()

                        #Adding the run matedata to the tensorboard summary writer
                        train_writer.add_run_metadata(run_metadata,'step%dbatch%d'%(i,bno))

                        #Creating the Timeline object and saving it to the json
                        tline=timeline.Timeline(run_metadata.step_stats)
                        #Creating the chrome trace
                        ctf=tline.generate_chrome_trace_format()
                        timeline_path=timeline_filename+'timeline_step%dbatch%d.json'%(i,bno)
                        with open(timeline_path,'w') as f:
                            f.write(ctf)

                    else:
                        #Starting the timer
                        t0=datetime.datetime.now()
                        #Running the op to train
                        track_results=sess.run(train_track_ops,
                                                feed_dict={is_training:True},
                                                )
                        t1=datetime.datetime.now()

                    print 'Training loss @epoch: ',i,' @minibatch: ',bno,track_results[1:-1],'in ',t1-t0
                    #Now the last op has the merged_summary evaluated.So, write it.
                    train_writer.add_summary(track_results[-1],bno)
                    bno+=1
                except tf.errors.OutOfRangeError:
                    t_epoch_end=datetime.datetime.now()
                    print 'Training one epoch completed in: {}\n'.format(
                                    t_epoch_end-t_epoch_start)
                    break

            #get the validation accuracy,starting the validation/test iterator
            sess.run(test_iter_init_op)
            bno=1
            while i%6==0:
                try:
                    #_,datay=sess.run(next_element)#dont use iterator now
                    #print datay
                    #Running the op
                    t0=datetime.datetime.now()
                    #Run the summary also for the validation set.just leave the train op
                    track_results=sess.run(train_track_ops[1:],
                                            feed_dict={is_training:False},
                                            )
                    t1=datetime.datetime.now()
                    print 'Testing loss @epoch: ',i,' @minibatch: ',bno,track_results[0:-1],'in ',t1-t0
                    #Again write the evaluated summary to file
                    test_writer.add_summary(track_results[-1],bno)
                    bno+=1
                except tf.errors.OutOfRangeError:
                    print 'Validation check completed!!\n'
                    break

            #Also save the checkpoints (after two every epoch)
            if i%6==0:
                #Saving the checkpoints
                checkpoint_path=checkpoint_filename+'model.ckpt'
                saver.save(sess,checkpoint_path,global_step=i)


if __name__=='__main__':
    import optparse
    usage='usage: %prog[options]'
    parser=optparse.OptionParser(usage)
    parser.add_option('--data_dir',dest='data_dir',
                        help='Directory containing data',
                        default=local_directory_path)
    (opt,args)=parser.parse_args()
    local_directory_path=opt.data_dir
    if local_directory_path[-1] != '/':
        local_directory_path=local_directory_path+'/'

    #Setting up the name of the filelist of train and test dataset
    #Making the filelist for the train dataset
    train_filename_pattern=local_directory_path+'event_file_1_*zside_0.tfrecords'
    #Making the filelist for the test datasets
    test_filename_pattern=local_directory_path+'event_file_2_start_0*zside_0.tfrecords'


    #Seting up some metric of dataset and training iteration
    mini_batch_size=20
    buffer_size=mini_batch_size*2
    epochs=31

    #Setting up the learning rate Hyperparameter
    init_learning_rate=0.0009    #0.001 default for Adam
    decay_step=150
    decay_rate=0.90

    #parse_tfrecords_file(train_filename_list,test_filename_list,mini_batch_size)
    train(epochs,
            mini_batch_size,buffer_size,
            init_learning_rate,decay_step,decay_rate,
            train_filename_pattern,
            test_filename_pattern,
            restore_epoch_number=None)
