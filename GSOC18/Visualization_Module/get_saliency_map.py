import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

#Importing the required moduels from the CNN_module
import sys,os
sys.path.append(
        os.path.join(os.path.dirname(sys.path[0]),'CNN_Module'))
#Importing the models on which it was trained
from model1_definition import model6 as model_function_handle
#This inference pipeline could be used here also
from io_pipeline import parse_tfrecords_file_inference
#inporting the helper funtion from the inference module
from inference_multi_gpu import get_available_gpus

#Retreiving the checkpoints
run_number=32
checkpoint_filename=os.path.join(
            os.path.dirname(sys.path[0]),'CNN_Module/')+\
            'tmp/hgcal/{}/checkpoint/'.format(run_number)
#Giving the dataset location
local_directory_path=os.path.join(
            os.path.dirname(sys.path[0]),'GeometryUtilities-master/interpolation/image_data/')

###################### GRAPH CREATION ######################
def create_computation_graph(iterator,is_training,map_dimension):
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

def get_gradient(infer_filename_pattern,checkpoint_epoch_number,map_dimension):
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
    #Setting up the inference config of the is_training
    is_training=tf.constant(False,dtype=tf.bool,name='training_flag')

    #Creating the saliency map of only one element at a time
    mini_batch_size=1
    with tf.device('/cpu:0'):
        os_iterator=parse_tfrecords_file_inference(infer_filename_pattern,
                                                    mini_batch_size)

    saliency_ops=create_computation_graph(os_iterator,is_training,map_dimension)

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
        results_filename=os.path.join(
                    os.path.dirname(sys.path[0]),'CNN_Module/')+\
                    'tmp/hgcal/{}/results/'.format(run_number)+\
                    'saliency_map_arrays'
        np.savez_compressed(results_filename,
                            gradient=gradient,
                            input=input,
                            pred=pred,
                            label=label)
        print 'Saved the saliency map creation data at: ',results_filename

def create_layerwise_saliency_map(input_img,gradient):
    '''
    DESCRIPTION:
        This function creates the saliency map finally along side the
        layer hits for all the layers.
    USAGE:
        INPUT:
            input_img  : the input image to the CNN Module
            gradient   : the gradient of the map_dimension on the image
                        space
        OUTPUT:

    '''
    #Removing the unnecessary batch dimension which will be 1
    input_img=np.squeeze(input_img)
    gradient=np.squeeze(gradient)

    #Now plotting the image layer wise
    layers=input_img.shape[2]
    for i in range(layers):
        #Plotting the image and corresponding gradient
        fig=plt.figure()
        fig.suptitle('Image and corresponding Saliency Map(gradient)')

        #Making the image axes
        ax1=fig.add_subplot(121)
        # x=range(input_img.shape[1])
        # y=range(input_img.shape[0])
        # xx,yy=np.meshgrid(x,y)

        image=input_img[:,:,i]
        #ax1.imshow(image,cmap='Dark2',interpolation='nearest')
        #Trying the 3d Surface PLot
        # print xx.shape,yy.shape,image.shape
        # ax1.plot_surface(xx,yy,image)
        #Trying the scatter plot
        x=[]
        y=[]
        for ix in range(image.shape[0]):
            for jy in range(image.shape[1]):
                if not image[ix,jy]==0:
                    y.append(ix)
                    x.append(jy)
        ax1.scatter(x,y,alpha=0.25)
        ax1.set_xlim(0,514)
        ax1.set_ylim(514,0)

        ax2=fig.add_subplot(122)
        map=gradient[:,:,i]
        ax2.imshow(map,cmap='jet',interpolation='nearest')

        plt.show()
        #plt.close()


def create_3d_scatter_saliency_map(input_img,gradient):
    '''
    DESCRIPTION:
        This function will try to visualize the gradient and hits as
        a 3d scatter plot.
        This will give us the global view of the positional location
        of the hit and the important region in 3d scatter plot.

        This will loose the relative importance (could be colored later)
        but show us the global 3d view of hits and important region.
    USAGE:
        INPUT:
            input    : the input "image" to the CNN
            gradient : the gradient with respect to the map dimension
                        in the image space
    '''
    #Squezzing the inputs
    input_img=np.squeeze(input_img)
    gradient=np.squeeze(gradient)

    #Starting the plot
    fig=plt.figure()
    fig.suptitle('3D scater of hits and it imporance in prediction')

    #Plottig the hits
    ax1=fig.add_subplot(121,projection='3d')
    x=[]
    y=[]
    z=[]
    for ix in range(input_img.shape[0]):
        for iy in range(input_img.shape[1]):
            for layer in range(input_img.shape[2]):
                if not input_img[ix,iy,layer]==0:
                    x.append(ix)
                    y.append(iy)
                    z.append(layer)
    ax1.scatter(x,y,z)

    #plotting the important region
    ax2=fig.add_subplot(122,projection='3d')
    x2=[]
    y2=[]
    z2=[]
    for ix in range(gradient.shape[0]):
        for iy in range(gradient.shape[1]):
            for layer in range(gradient.shape[2]):
                if not gradient[ix,iy,layer]==0:
                    x2.append(ix)
                    y2.append(iy)
                    z2.append(gradient[ix,iy,layer])
    ax2.scatter(x2,y2,z2)


    plt.show()



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
        create_layerwise_saliency_map(input,gradient)

        #Creating the scatter 3d representation of the saliency map
        #create_3d_scatter_saliency_map(input,gradient)
