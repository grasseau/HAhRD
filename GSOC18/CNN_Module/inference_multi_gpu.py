import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import datetime
import os
from io_pipeline import parse_tfrecords_file_inference


################# GLOBAL VARIABLES #####################
#Getting the model handle
from model1_definition import model5 as model_function_handle
from model1_definition import calculate_total_loss,calculate_model_accuracy
#model_function_handle=model2
#default directory path for datasets
local_directory_path='/home/gridcl/kumar/HAhRD/GSOC18/GeometryUtilities-master/interpolation/image_data'
#Checkpoint file path
run_number=20
checkpoint_filename='tmp/hgcal/{}/checkpoint/'.format(run_number)
#Directory to save the prediction in compressed numpy format
results_basepath='tmp/hgcal/{}/results/'.format(run_number)
if not os.path.exists(results_basepath):
    os.mkdir(results_basepath)



################# HELPER FUNCTIONS #####################
def _get_available_gpus():
    '''
    DESCRIPTION:
        This function is same as that used in train_multi_gpu
        script. One modification that could be done later is
        to run the inference on the system which dont have any
        GPU but instead just CPUs.
        So this function could return those names also instead.

    USAGE:
        OUTPUT:
            all_gpu_name    :list of name of of all the gpus which
                                are visible to tensorflow.
    '''
    #This will give the list of all the devices (including CPUs)
    local_devices=device_lib.list_local_devices()

    #Now filtering the GPU devices to run the inference.
    '''Test whether running inference affect with different devices
    since batchnorm statistics will be saved, which will be specific
    to the devices. So atleast we need to have same graph to run
    the inference after restoring the checkpoint? unless all the
    weights (including the BNs were on cpu)'''
    all_gpu_name=[x.name for x in local_devices
                                if x.device_type=='GPU']

    return all_gpu_name

#################### MAIN FUCTIONS DEFINITIONS ###################
def create_inference_graph(iterator,is_training):
    '''
    DESCRIPTION:
        This function will create a similar graph as that used during
        training, and use the restored variables saved in the
        checkpoint later.
    USAGE:
        INPUT:
            iterator    : the iterator to fetch the next batch of
                            data for inference.
            is_training : the boolean flag to tell that we are in
                            inference phase and to stop the dropout
                            and used appropriate BN statistics.
        OUTPUT:
            label_pred_ops : the list of containing the prediction
                                from each of the tower like:
                                [Z1,Z2,....]
                                (losses could also be included later)
    '''
    all_gpu_name=_get_available_gpus()
    num_gpus=len(all_gpu_name)
    label_pred_ops=[]
    accuracy_ops=[]

    with tf.variable_scope(tf.get_variable_scope()):
        #Launching the graph one by one on each device
        for i in range(num_gpus):
            #Creating the device scope of this GPU to create graph on
            with tf.device(all_gpu_name[i]):
                #Creating the name scope to distinguish the statistics
                #of this device (for for naming purpose)
                with tf.name_scope('tower{}'.format(i)) as tower_scope:
                    #Fetching the (image,target) batch for prediction
                    ((X,_),(Y,_))=iterator.get_next()

                    #Now,making the prediction using the model
                    Z=model_function_handle(X,is_training)
                    total_cost=calculate_total_loss(Z,Y,tower_scope)
                    accuracy_tuple=calculate_model_accuracy(Z,Y)

                    #Appending the label and prediction with loss for verification
                    label_pred_ops.append((Y,Z,total_cost))
                    accuracy_ops.append(accuracy_tuple)

                    #Making the varible resuse in this variable scope
                    #i.e ultimately having a master copy of variable on cpu
                    tf.get_variable_scope().reuse_variables()

    return label_pred_ops,accuracy_ops

def infer(test_image_filename_list,test_label_filename_list,
            inference_mode,mini_batch_size,checkpoint_epoch_number):
    '''
    DESCRIPTION:
        This function will now control the whole inference process
        1. Making the one-shot-iterator of the test dataset,
        2. Creating the minimal inference graph
        3. Loading the saved-weights from the checkpoints
        4. Running the seesion to get all the training and prediction
        5. Finally calculating the params like diff,error,
        6. Calling the plotting and other functions for visualization
    USAGE:
        INPUT:
            test_image_filename_list : the filename list of the tfrecords
                                        of the test "images".
            test_label_filename_list : the filename list for the tfrecords
                                        of the test labels
            inference_mode           : to specify whether we are infering
                                        on valid/train mode. (will be used
                                        just for naming purpose)
            mini_batch_size          : the size of image to process parallely
            checkpoint_epoch_number  : the checkpoint number of the file
                                        saved at that epoch

    '''
    #Setting up the required config i.e mode of the running the graph
    is_training=tf.constant(False,dtype=tf.bool,name='training_flag')

    #Getting the one-shot-iterator of the testing dataset
    with tf.device('/cpu:0'):
        os_iterator=parse_tfrecords_file_inference(test_image_filename_list,
                                                test_label_filename_list,
                                                mini_batch_size)

    #Creating the graph for inference
    label_pred_ops,accuracy_ops=create_inference_graph(os_iterator,is_training)
    #Initializing the result array
    predictions=None
    labels=None
    accuracies=None

    #Starting the saver to load the checkpoints
    saver=tf.train.Saver()

    config=tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    with tf.Session(config=config) as sess:
        #Directily initializing the variables with the checkpoint file
        checkpoint_path=checkpoint_filename+'model.ckpt-{}'.format(
                                                checkpoint_epoch_number)
        #Restoring with the saver
        print '>>>> Restoring Model from saved checkpoint at: ',checkpoint_path
        saver.restore(sess,checkpoint_path)

        #Now running the inference
        bno=1
        while True:
            #Iterating till the one-shot-iterator get exhausted
            try:
                print '\n\n>>>Making inference for the batch: {}'.format(bno)
                #running the inference op (phase: testing automatically given)
                t0=datetime.datetime.now()
                #Running both inference and accuracy in one run. (IMPORTANT)
                infer_results,accuracy_results=sess.run([label_pred_ops,accuracy_ops])

                #Unzipping the target,prediction and losses of each tower
                tower_labels,tower_predictions,tower_losses=zip(*infer_results)
                #Converting the accuracies tuples of all the tower into numpy array
                accuracy_results=[np.reshape(np.array(ac_tup),(1,-1))
                                        for ac_tup in accuracy_results]

                #Making appropriate arrays from the resluts of ops
                if bno==1:
                    #making the label-prediction array
                    labels=np.concatenate(tower_labels,axis=0)
                    predictions=np.concatenate(tower_predictions,axis=0)
                    #Concatenating the accuracies in one row
                    accuracies=np.concatenate(accuracy_results,axis=0)
                else:
                    #Joining the predictions along the batch axis to make one big result
                    labels=np.concatenate([labels]+list(tower_labels),axis=0)
                    predictions=np.concatenate([predictions]+list(tower_predictions),axis=0)
                    #Concatenating the accuracy results to the accuracy array
                    accuracies=np.concatenate([accuracies]+accuracy_results,axis=0)

                t1=datetime.datetime.now()
                print 'loss of this minibatch: ',tower_losses
                print 'predictions shape: ',predictions.shape,labels.shape
                print 'accuracies shape: ',accuracies.shape
                print 'Inference for batch completed in: ',t1-t0,'\n'
                bno+=1

            except tf.errors.OutOfRangeError:
                print '>>>>Inference of Test Dataset Complete'
                break

    #Saving the numpy array in compresed format
    print '>>>>Saving the prediction in ',results_basepath
    results_filename=results_basepath+'results_mode_{}'.format(inference_mode)
    np.savez_compressed(results_filename,
                        predictions=predictions,
                        labels=labels)

    #Printing the Error/Accracies collected in accuracies variable
    average_accuracies=np.mean(accuracies,axis=0)
    print '>>>>Error/Accuracies in order:\n\n\n ',average_accuracies



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

    #Making the prediction on the Training Set
    #Setting up the train data directory
    train_image_filename_list=[local_directory_path+'image0batchsize1000zside0.tfrecords']
    train_label_filename_list=[local_directory_path+'label0batchsize1000.tfrecords']
    #Making inference
    infer(train_image_filename_list,
            train_label_filename_list,
            inference_mode='train',
            mini_batch_size=10,
            checkpoint_epoch_number=30)

    #Now resetting the tf graph to make a new infrence on test image dataset
    print '>>>>>Resetting the default graph\n\n'
    tf.reset_default_graph()

    #Making the prediction on Test Set
    #Setting the name of the test data directory
    test_image_filename_list=[local_directory_path+'image1000batchsize1000zside0.tfrecords']
    test_label_filename_list=[local_directory_path+'label1000batchsize1000.tfrecords']

    infer(test_image_filename_list,
        test_label_filename_list,
        inference_mode='valid',
        mini_batch_size=10,
        checkpoint_epoch_number=30)
