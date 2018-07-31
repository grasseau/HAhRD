import tensorflow as tf
import numpy as np
import datetime
import sys,os

################# CUSTOM SCRIPT IMPORT ####################
#Importing the required moduels from the CNN_module
# sys.path.append(
#         os.path.join(os.path.dirname(sys.path[0]),'CNN_Module'))
#Importing the models on which it was trained
#from model1_definition import model6 as model_function_handle
#This inference pipeline could be used here also
from CNN_Module.utils.io_pipeline import parse_tfrecords_file_inference

################### SAVE and RETREIVE DIRECTORY HANDLING ####
#Retreiving the checkpoints
# run_number=32
# checkpoint_filename=os.path.join(
#             os.path.dirname(sys.path[0]),'CNN_Module/')+\
#             'tmp/hgcal/{}/checkpoint/'.format(run_number)
#Giving the dataset location
# local_directory_path=os.path.join(
#             os.path.dirname(sys.path[0]),'GeometryUtilities-master/interpolation/image_data/')

#Save directory
# save_dir='saliency_map_plots/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

###########################################################
###################### GRAPH CREATION ######################
def get_available_gpus():
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

def create_computation_graph(model_function_handle,
                    iterator,is_training,map_dimension):
    '''
    DESCRIPTION:
        This function is similar to the trainign graph but with only
        the minimal ops necessary to create this saliency maps.
    USAGE:
        INPUT:
            iterator    : the input image will be  received from the iterator
                            (one shot iterator)
            is_training : the flag which will be used by the dropout and batchnorm
                            to know whether we are in training or testing to
                            use appropriate BN statistics and stop dropout.
            map_dimension: the dimension in the output of which we have to
                            calculate the saliency map to input space
        OUTPUT:

    '''
    all_gpu_name=get_available_gpus()
    num_gpus=len(all_gpu_name)

    with tf.variable_scope(tf.get_variable_scope()):
        #Automatically place the ops where ever is fine
        for i in range(num_gpus):
            with tf.device(all_gpu_name[i]):
                with tf.name_scope('tower%s'%(i)) as tower_scope:
                #Getting the inference image
                    X,Y=iterator.get_next()
                    #Running the foreward propagation op
                    Z=model_function_handle(X,is_training)
                    #Getting the gradient with respect to the input
                    gradient=tf.gradients(Z[0,map_dimension],#y
                                        X,#x
                                        )#get dy/dx

                    tf.get_variable_scope().reuse_variables()


    return [gradient,X,Z,Y]

def get_gradient(run_number,
                model_function_handle,
                infer_filename_pattern,
                checkpoint_epoch_number,
                map_dimension):
    '''
    DESCRIPTION:
        This function will control all the tensorflow relate ops and
        finally return the gradient to be visualized as the saliency
        map.
    USAGE:
        INPUT:
            infer_filename_pattern  : the filename patter on which
                                        we have to do this new inference
            checkpoint_epoch_number : the checkpoint number which we want to
                                        restore
            map_dimension           : the dimension of output where we want
                                        to create the saliency map
        OUTPUT:

    '''
    #Setting up the necessary file and folders for checkpoints and saving
    #Setting the checkpoint file directory for loading it
    checkpoint_filename='tmp/hgcal/{}/checkpoint/'.format(run_number)
    #Directory to save the saliency maps
    save_dir='tmp/hgcal/{}/results/'.format(run_number)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #Setting up the inference config of the is_training
    is_training=tf.constant(False,dtype=tf.bool,name='training_flag')

    #Creating the saliency map of only one element at a time
    mini_batch_size=1
    with tf.device('/cpu:0'):
        os_iterator=parse_tfrecords_file_inference(infer_filename_pattern,
                                                    mini_batch_size)

    saliency_ops=create_computation_graph(model_function_handle,
                        os_iterator,is_training,map_dimension)

    #Getting the saver handle
    saver=tf.train.Saver()

    config=tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    with tf.Session(config=config) as sess:
        #Restoring the checkpoint
        checkpoint_path=checkpoint_filename+'model.ckpt-{}'.format(
                                                checkpoint_epoch_number)
        print '>>> Restoring the model from the saved checkpoint at: ',checkpoint_path
        saver.restore(sess,checkpoint_path)
        print '>>> Checkpoint Restored'

        #Getting the gradient op and other useful data
        #(run it in bigger batch cuz it takes lot of time to use tf.gradient
        #see the answer by mmry)
        gradient,input,pred,label=sess.run(saliency_ops)
        print 'Gradient Shape: ',gradient[0].shape
        print 'Label: ',label
        print 'Prediction: ',pred

        print '>>> Saving the data for map creation in results folder'
        results_filename=save_dir+'saliency_map_arrays'
        np.savez_compressed(results_filename,
                            gradient=gradient,
                            input=input,
                            pred=pred,
                            label=label)
        print 'Saved the saliency map creation data at: ',results_filename


if __name__=='__main__':
    import optparse
    usage='usage: %prog[option]'
    parser=optparse.OptionParser(usage)
    parser.add_option('--mode',dest='mode',
                     help='create or visual mode',
                    default='')
    (opt,args)=parser.parse_args()

    if opt.mode=='create':
        #Setting up the data and checkpoint file
        checkpoint_epoch_number=30
        map_dimension=0
        infer_filename_pattern=local_directory_path+\
                    'event_file_1*zside_1.tfrecords'

        #Calling the function to get the gradient
        get_gradient(infer_filename_pattern,
                                checkpoint_epoch_number,
                                map_dimension)
    else:
        #Loading the data from the npz file
        filename='saliency_map_arrays.npz'
        #Getting the required variable
        data=np.load(filename)
        gradient=data['gradient']
        input=data['input']
        pred=data['pred']
        label=data['label']

        print 'label: ',label
        print 'pred: ',pred

        #Calling the plotting function
        create_layerwise_saliency_map_matplot(input,gradient)

        #Creating the scatter 3d representation of the saliency map
        #create_3d_scatter_saliency_map(input,gradient)

        #Calling the plotly plot
        create_layerwise_saliency_map(input,gradient)
