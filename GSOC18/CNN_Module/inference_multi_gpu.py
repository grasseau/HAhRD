import tensorflow as tf
from tensorflow.python.client import device_lib


################# GLOBAL VARIABLES #####################
#Getting the model handle
from model1_definition import model2
model_function_handle=model2
#Checkpoint file path
checkpoint_filename='tmp/hgcal/checkpoint/'


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

                    #Appending the label and prediction
                    label_pred_ops.append((Y,Z))

                    #Making the varible resuse in this variable scope
                    #i.e ultimately having a master copy of variable on cpu
                    tf.get_variable_scope().reuse_variables()

    return label_pred_ops

def infer(test_image_filename_list,test_label_filename_list,
            mini_batch_size,checkpoint_epoch_number):
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
            mini_batch_size          : the size of image to process parallely
            checkpoint_epoch_number  : the checkpoint number of the file
                                        saved at that epoch

    '''
    #Setting up the required config i.e mode of the running the graph
    is_training=tf.constant(False,dtype=tf.bool,name='training_flag')

    #Getting the one-shot-iterator of the testing dataset
    os_iterator=parse_tf_records_file_inference(test_image_filename_list,
                                                test_label_filename_list,
                                                mini_batch_size)

    #Creating the graph for inference
    label_pred_ops=create_inference_graph(iterator,is_training)
    #Initializing the result array
    results=None

    with tf.Session() as sess:
        #Directily initializing the variables with the checkpoint file
        checkpoint_path=checkpoint_filename+'model.ckpt-{}'.format(
                                                checkpoint_epoch_number)
        #Restoring with the saver
        saver.restore(sess,checkpoint_path)

        #Now running the inference
        bno=1
        while True:
            #Iterating till the one-shot-iterator get exhausted
            try:
                print 'Making inference for the batch: {}'.format(bno)
                bno+=1
                #running the inference op (phase: testing automatically given)
                infer_results=sess.run(label_pred_ops)
                if results==None:
                    results=infer_results
                else:
                    #Joining the results along the batch axis to make one big result
                    results=np.concatenate((results,infer_results),axis=0)

            except tf.errors.OutOfRangeError:
                print 'Inference of Test Dataset Complete'
                break

    #Now Plotting the results and other furthur visualization of results
    #plot_results_visualization(results)

def plot_results_visualization(results):
    '''
    DESCRIPTION:
        This function will visualize the results by plotting:
        1.Prediction
    '''

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

    #Setting the name of the test data directory
    test_image_filename_list=[local_directory_path+'image1000batchsize1000zside0.tfrecords']
    test_label_filename_list=[local_directory_path+'label1000batchsize1000.tfrecords']

    infer(test_image_filename_list,
        test_label_filename_list,
        mini_batch_size=10,
        checkpoint_epoch_number=30)
    
