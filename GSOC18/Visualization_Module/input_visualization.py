#Importing required visualization modules
import plotly
import plotly.graph_objs as go
import numpy as np

#importing the helper functions
from config import *
from visualization_io import *

##################### GLOBAL VARIABLES #######################
save_dir='animation/'       #the folder in current directory to save


def create_event_animation(layers,event_image,
                            dataformat='HWC',name='kalpana'):
    '''
    DESCRIPTION:
        This function will create a layer wise animation of the
        hits in the event "image" file.

        In Progress:
        Currently it is based on the MRI-animation but we plan to
        include the trace of the hits as the layers goes back upto
        Hadronic Calorimeter(last layer).
    USAGE:
        INPUT:
            layers      : the total number of layers in the event
            event_image : the image interpolated from the hits in
                            the event in [C,H,W] format
            name        : the name for the animation that will be
                            saved. defaulted to "imagination"
        OUTPUT:
            html_file   : an auto generated animation file will be
                            saved in the directory
    '''
    #Transforming the data to suit our style of format (HWC)
    print '>>> Transforming the data into correct shape'
    assert (dataformat=='HWC' or
        dataformat=='CHW'),'Please give in valid format (HWC/CHW)'
    if dataformat=='CHW':
        event_image=np.tanspose(event_image,(1,2,0))
    print 'shape: {}'.format(event_image.shape)

    print '>>> Creating the plots'
    #Getting the colorscale
    colorscale=get_colorscale_view_energy()

    ####################### DATA ##############################
    #defining the data variable (default to zeroth layer)
    data=[
        go.Surface(
            z=0+event_image[:,:,0]*0,
            surfacecolor=event_image[:,:,0],
            colorbar=colorbar,
            colorscale=colorscale,
            text=z[:,:,0]           #for hovertext with the energy
        )
    ]

    ##################### FRAME ################################
    #defining the frames for animating
    frames=[
        {
            "data":[
                go.Surface(
                    z=i+z[:,:,i]*0,
                    surfacecolor=event_image[:,:,i],
                    colorbar=colorbar,
                    colorscale=colorscale,
                    text=z[:,:,i] #dont use with hoverinfor.
                    #the x,y values formatting looks bad
                )
            ],
            "name":'frame{}'.format(i+1),
        }
        for i in range(layers)
    ]

    #Creating the steps for the sliders to slide the frames
    frame_duration=500
    transition_duration=300
    slider_steps=[]
    for i in range(layers):
        step=dict(
            label='{}'.format(i+1),
            method='animate',
            args=[
                ['frame{}'.format(i+1)],   #to associate a frame
                dict(
                    frame={'duration':frame_duration,'redraw':True},
                    mode='immediate',
                    transition={'duration':transition_duration},
                )
            ]
        )
        slider_steps.append(step)
    sliders[0]['steps']=slider_steps


    ##################### LAYOUT #############################
    #defining the layout of the plot
    #Getting the scene to set the axes
    scene=get_scene(event_image.shape)
    layout=dict(
        title='Event Hit Interactive Plot',
        scene=scene,
        margin=margin,
        updatemenus=updatemenus,
        sliders=sliders,
    )

    #Generating the figure
    fig=dict(
        data=data,
        layout=layout,
        frames=frames,
        config={
            'scrollzoom':True
        }
    )
    #Finally plotting the plot which will automatically save
    plotly.offline.plot(fig,filename=name+'.html')

if __name__=='__main__':
    # filename="image0batchsize20.tfrecords.npy"
    # Z=np.load(filename)
    # nx,ny=(514,513)
    # z=Z[3,:,:,:]
    # print np.max(z),z.shape,z[:,:,0]

    filename_list=['image0batchsize20.tfrecords']
    next_element=get_tf_records_iterator(filename_list)

    sess=tf.Session()
    z=image=sess.run(next_element)

    print type(image)
    print image.shape


    layers=40
    create_event_animation(layers,z)
