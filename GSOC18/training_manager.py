import tensorflow as tf

#Adding the default path to the data directory
dataset_directory='GeometryUtilities-master/interpolation/image_data/'

#importing the model to be used for training
from models/model1_definition import model7 as model_function_handle
from models/model1_definition import calculate_model_accuracy
from models/model1_definition import calculate_total_loss

#import the trainer and inference functions
from train_multi_gpu import train
from inference_multi_gpu import infer

###################### RUN CONFIGURATION #####################
run_number=35
#the regex pattern for the dataset filename
train_filename_pattern='event_file_1_*zside_0.tfrecords'
test_filename_pattern='event_file_2_*zside_0.tfrecords'

if __name__=='__main__':

    #Parsing the arguments
    import optparse
    usage='usage: %prog[options]'
    parser=optparse.OptionParser(usage)
    parser.add_option('--mode',dest='mode',
                        help='train/infer/hparam_search')
    parser.add_option('--dataset_directory',dest='dataset_directory',
                        help='directory with the dataset',
                        default=dataset_directory)
    (opt,args)=parser.parse_args()

    #Correcting the dataset directory if / is not specified at end
    dataset_directory=opt.dataset_directory
    if dataset_directory[-1] != '/':
        dataset_directory=dataset_directory+'/'
    #Finally giving the full path to the dataset
    train_filename_pattern=dataset_directory+train_filename_pattern
    test_filename_pattern=dataset_directory+test_filename_pattern


    ################## TRAINING HANDLE ####################
    '''
    DESCRIPTION:
        This is the main control of all the training and the hyperparameter
        definition of the training.
        All the tensorboard visualization and the checkpoints of the model
        will be saved in the current directory under the directory structure:

        current_directory/tmp/hgcal/run_number

        This direcotry is by default added to the gitignore
    '''
    if opt.mode=='train':
        #Specifying the Hyperparameters
        init_learning_rate=0.001
        decay_step=150
        decay_rate=0.95
        #Specifying the run configuration
        mini_batch_size=20
        shuffle_buffer_size=mini_batch_size*2 #for shuffling the dataset files
        epochs=31
        restore_epoch_number=None

        #Finally running the training
        train(run_number,
                model_function_handle,
                calculate_model_accuracy,
                calculate_total_loss,
                epochs,mini_batch_size,shuffle_buffer_size
                init_learning_rate,decay_step,decay_rate,
                train_filename_pattern,test_filename_pattern,
                restore_epoch_number=restore_epoch_number)

    ############## INFERENCE HANDLE #######################
    '''
    DESCRIPTION:
        This is the main point of control for all the inferece task like
        calulating the average accuracy on the training and test/dev set
        for furthur visualization.

        All the results from the current inference will be saved in same
        directory strucute as train module:

        current_directory/tmp/hgcal/run_number
    '''
    #specifying the inference configuration
    mini_batch_size=20
    checkpoint_epoch_number=30

    #Running the inference on the training data set
    infer(run_number,
            model_function_handle,
            calculate_model_accuracy,
            calculate_total_loss,
            train_filename_pattern,
            inference_mode='train',#on the training dataset
            mini_batch_size=mini_batch_size,
            checkpoint_epoch_number=checkpoint_epoch_number)

    #Now rerunning the inference on the test dataset
    tf.reset_default_graph()
    infer(run_number,
            model_function_handle,
            calculate_model_accuracy,
            calculate_total_loss,
            test_filename_pattern,
            inference_mode='valid',
            mini_batch_size=mini_batch_size,
            checkpoint_epoch_number=checkpoint_epoch_number)
