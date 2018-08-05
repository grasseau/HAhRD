import tensorflow as tf
import numpy as np

#Adding the default path to the data directory
default_dataset_directory='GeometryUtilities-master/interpolation/image_data/'

#importing the model to be used for training
from models.model_rnn_definition import model9 as model_function_handle
from models.model1_definition import calculate_model_accuracy
from models.model_rnn_definition import calculate_total_loss

#import the trainer and inference functions
from train_multi_gpu import train
from inference_multi_gpu import infer
#import the gradient calulation function
from get_saliency_map import get_gradient

###################### RUN CONFIGURATION #####################
run_number=44
#the regex pattern for the dataset filename
train_filename_pattern='nopu/valid/*'
test_filename_pattern='nopu/valid/*'
test_pu_filename_pattern='test_pu/*'
viz_filename_pattern='test_pu/*'

if __name__=='__main__':

    #Parsing the arguments
    import optparse
    usage='usage: %prog[options]'
    parser=optparse.OptionParser(usage)
    parser.add_option('--mode',dest='mode',
                        help='train/infer/hparam_search')
    parser.add_option('--dataset_directory',dest='dataset_directory',
                        help='directory with the dataset',
                        default=default_dataset_directory)
    (opt,args)=parser.parse_args()

    #Correcting the dataset directory if / is not specified at end
    dataset_directory=opt.dataset_directory
    if dataset_directory[-1] != '/':
        dataset_directory=dataset_directory+'/'
    #Finally giving the full path to the dataset
    train_filename_pattern=dataset_directory+train_filename_pattern
    test_filename_pattern=dataset_directory+test_filename_pattern
    test_pu_filename_pattern=dataset_directory+test_pu_filename_pattern
    viz_filename_pattern=dataset_directory+viz_filename_pattern


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
        decay_step=2500
        decay_rate=0.95
        #Specifying the run configuration
        mini_batch_size=10
        shuffle_buffer_size=mini_batch_size*2 #for shuffling the dataset files
        epochs=31
        restore_epoch_number=None

        #Finally running the training
        train(run_number,
                model_function_handle,
                calculate_model_accuracy,
                calculate_total_loss,
                epochs,mini_batch_size,shuffle_buffer_size,
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
    if opt.mode=='infer':
        mini_batch_size=20
        checkpoint_epoch_number=31

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

        #Now running the inference on the PU dataset
        # tf.reset_default_graph()
        # infer(run_number,
        #         model_function_handle,
        #         calculate_model_accuracy,
        #         calculate_total_loss,
        #         test_pu_filename_pattern,
        #         inference_mode='test_pu',
        #         mini_batch_size=mini_batch_size,
        #         checkpoint_epoch_number=checkpoint_epoch_number)

    ############# Visulaization Handle ###############
    '''
    Description:
        This will be the main point of control for making all the
        visualization of the training including the currently developed
        visualizations like
            1. prediction visulaization (includes error histograms)
            2. saliency maps (the gradient maps giving the sensitive
                                regions of the prediction in image)
            other visualization will be added later

        This manager will use the results saved by the inference module
        in tmp/hgcal/run_number/results to make the visualization
    '''
    if opt.mode=='pred_viz':
        #Inmporting the modules for visualization process
        from Visualization_Module.prediction_visualization import plot_histogram,load_data
        #################### Histogram Plots ###################
        #Loading the data
        filename='tmp/hgcal/{}/results/results_mode_train.npz'.format(run_number)
        train_results=load_data(filename)
        predictions=train_results['predictions']
        labels=train_results['labels']
        #Plotting the histogram
        plot_histogram(predictions,labels)

        #plotting the test set prediction histograms
        filename='tmp/hgcal/{}/results/results_mode_test_pu.npz'.format(run_number)
        test_results=load_data(filename)
        predictions=test_results['predictions']
        labels=test_results['labels']
        #Plotting the histogram
        plot_histogram(predictions,labels)

    if opt.mode=='map_gen':
        #################### Saliency Map #####################
        #Creating the saliency map
        checkpoint_epoch_number=9
        #Choosing wrt which output dimension we want to calculate gradient
        map_dimension=0
        #Number of images we want to process in parallel
        '''
        Currently only minibatch size of 1 is supported for calculation of
        gradient. For more information follow this thread.
        https://github.com/tensorflow/tensorflow/issues/4897
        '''
        mini_batch_size=1
        filename='test_pu'
        #Calulating the gradient
        get_gradient(run_number,
                    model_function_handle,
                    viz_filename_pattern,
                    mini_batch_size,
                    checkpoint_epoch_number,
                    map_dimension,
                    filename)

    if opt.mode=='map_viz':
        #Importing the necessary modules
        from Visualization_Module.prediction_visualization import plot_histogram,load_data
        from Visualization_Module.saliency_map_visualization import create_layerwise_saliency_map
        from Visualization_Module.saliency_map_visualization import create_layerwise_saliency_map_matplot
        #Now visualizing the gradient
        #Loading the gradient data
        filename='tmp/hgcal/{}/results/saliency_map_arrays.npz'.format(run_number)
        map_data=load_data(filename)
        plot_example=0
        gradient=np.squeeze(map_data['gradient'])[plot_example,:,:,:]
        input=map_data['input'][plot_example,:,:,:]
        pred=map_data['pred'][plot_example,:]
        label=map_data['label'][plot_example,:]
        #creating the visualization
        create_layerwise_saliency_map_matplot(input,gradient*input)

        #Creating the saliency map
        print gradient.shape,input.shape
        print pred
        print label
        save_dir='tmp/hgcal/{}/results/'.format(run_number)
        create_layerwise_saliency_map(input,gradient*input,save_dir)
