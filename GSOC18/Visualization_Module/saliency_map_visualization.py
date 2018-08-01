import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
#Importing the plotly requirements
import plotly
import plotly.graph_objs as go


##################### VISUALIZATION #######################
def create_layerwise_saliency_map_matplot(input_img,gradient):
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
    # for i in range(layers):
    #     #Plotting the image and corresponding gradient
    #     fig=plt.figure()
    #     fig.suptitle('Image and corresponding Saliency Map(gradient)')
    #
    #     #Making the image axes
    #     ax1=fig.add_subplot(121)
    #     # x=range(input_img.shape[1])
    #     # y=range(input_img.shape[0])
    #     # xx,yy=np.meshgrid(x,y)
    #
    #     image=input_img[:,:,i]
    #     #ax1.imshow(image,cmap='Dark2',interpolation='nearest')
    #     #Trying the 3d Surface PLot
    #     # print xx.shape,yy.shape,image.shape
    #     # ax1.plot_surface(xx,yy,image)
    #     #Trying the scatter plot
    #     x=[]
    #     y=[]
    #     for ix in range(image.shape[0]):
    #         for jy in range(image.shape[1]):
    #             if not image[ix,jy]==0:
    #                 y.append(ix)
    #                 x.append(jy)
    #     ax1.scatter(x,y,alpha=0.25)
    #     ax1.set_xlim(0,514)
    #     ax1.set_ylim(514,0)
    #
    #     ax2=fig.add_subplot(122)
    #     map=gradient[:,:,i]
    #     ax2.imshow(map,cmap='jet',interpolation='nearest')
    #
    #     plt.show()
    #     #plt.close()

    #Adding the gradient to show per layer activation (i.e importance)
    all_hit_sum=[]
    all_gradient_sum=[]
    for layer in range(layers):
        hit_sum=np.sum(input_img[:,:,layer])
        gradient_sum=np.sum(gradient[:,:,layer])

        all_hit_sum.append(hit_sum)
        all_gradient_sum.append(gradient_sum)

    fig=plt.figure()
    fig.suptitle('Variation of Sum of Hits and Gradient with layer')

    #The hit plots
    ax1=fig.add_subplot(121)
    ax1.plot(all_hit_sum)
    ax1.set_title('Variation of Sum of Hits Energy with layer')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Sum of Ennergy of Hits')

    #The gradient plots
    ax2=fig.add_subplot(122)
    ax2.plot(all_gradient_sum)
    ax2.set_title('Sum of gradient with layer')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Sum of Gradient for Energy Prediction')

    plt.show()


def create_3d_scatter_saliency_map_matplot(input_img,gradient):
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


def create_layerwise_saliency_map(input_img,gradient,save_dir):
    '''
    DESCRIPTION:
        Now finally we will make the visualization of the saliency
        map in the plotly interface.
    USAGE:
        INPUT:
            input_img   : the input image to the CNN
    '''
    #Preprocessing the data
    input_img=np.squeeze(input_img)
    gradient=np.squeeze(gradient)
    layers=40

    #Creating the subplots
    fig=plotly.tools.make_subplots(
                                rows=1,
                                cols=2,
                                subplot_titles=('Hits Scatter Plot (marker size not to scale)',
                                                'Saliency Map(Gradient in image-space)')
    )

    ################### DATA ######################
    #Setting the image trace
    hit_x,hit_y=np.argwhere(input_img[:,:,0]!=0).T
    hit_color=[]
    for i in range(hit_x.shape[0]):
                hit_color.append(input_img[hit_x[i],hit_y[i],0])
    #Giving the initial state of the visualization
    image_trace=go.Scatter(
                    x=hit_x,
                    y=hit_y,
                    #name='Hit Scatter Plot',
                    mode='markers',
                    marker=dict(
                        color=hit_color,
                        showscale=True,
                        line=dict(width=1),
                        colorscale='Viridis',
                        colorbar=dict(
                                    x=0.42,
                                    # y=0.8,
                                    # len=0.8
                        )
                    ),
                    text=['energy: '+str(hit_color[i])
                                for i in range(len(hit_color))],
                    xaxis='x1',
                    yaxis='y1'
    )
    fig.append_trace(image_trace,1,1)
    #Giving the gradient trace
    gradient_trace=go.Heatmap(
                        z=gradient[:,:,0].T,#ot make it similar to scatter
                        xaxis='x2',
                        yaxis='y2',
                        colorscale='Viridis',
                        colorbar=dict(
                                    x=1.02,
                                    # y=0.8,
                                    # len=0.8
                        )
    )
    fig.append_trace(gradient_trace,1,2)

    ################### FRAME ############################
    #IMAGE FRAME
    #Collecting the data for image frame
    hit_x_all=[]
    hit_y_all=[]
    hit_color_all=[]
    for layer in range(layers):
        #Calculating the information for this layer
        hit_x,hit_y=np.argwhere(input_img[:,:,layer]!=0).T
        hit_color=[]
        for hit_i in range(hit_x.shape[0]):
            hit_color.append(input_img[hit_x[hit_i],hit_y[hit_i],layer])
        #Appending the data for this layer to the all list
        hit_x_all.append(hit_x)
        hit_y_all.append(hit_y)
        hit_color_all.append(hit_color)
    #Creating the  Frame
    img_grad_frames=[
        {
            "data":[
                #Defining the plot to go on the image side for this frame
                go.Scatter(
                    x=hit_x_all[layer_i],
                    y=hit_y_all[layer_i],
                    name='hits',
                    mode='markers',
                    marker=dict(
                        color=hit_color_all[layer_i],
                        showscale=True,
                        line=dict(width=1),
                        colorscale='Viridis',
                        colorbar=dict(
                                    x=0.42,
                                    # y=0.8,
                                    # len=0.8
                        )
                    ),
                    text=['energy: '+str((hit_color_all[layer_i])[j])
                                for j in range(len(hit_color_all[layer_i]))],
                    #hoverinfo='text',
                    xaxis='x1',
                    yaxis='y1',
                ),
                #Defining the plot to go in the gradient side for this frame
                go.Heatmap(
                    z=gradient[:,:,layer_i].T,
                    name='gradient',
                    xaxis='x2',
                    yaxis='y2',
                    colorscale='Viridis',
                    colorbar=dict(
                                x=1.02,
                                # y=0.8,
                                # len=0.8
                    ),
                )
            ],
            "name":'frame{}'.format(layer_i+1)
        }
        for layer_i in range(layers)
    ]

    #################### SLIDER #########################
    sliders=[{
        'active':0,
        #Positioning of the sliders
        'yanchor':'top',
        'xanchor':'left',
        'x':0,
        'y':0,
        #Padding and length of the sliders
        'pad':{'t':50,'b':10},
        'len':1,
        #For displaying the current status of slider
        'currentvalue':{
            'font':{'size':20},
            'prefix':'Layer: ',
            'visible': True,
            'xanchor':'left'
        },
        #For integrating the animation frames later
        'transition':{'duration':300,'easing':'cubic-in-out'},
        'steps':[]   #add the frame which we want to animate in steps
    }]
    frame_duration=500
    transition_duration=300
    slider_steps=[]
    for i in range(layers):
        step=dict(
                label='{}'.format(i+1),
                method='animate',
                args=[
                    ['frame{}'.format(i+1)],
                    dict(
                        frame={'duration':frame_duration,'redraw':True},
                        mode='immediate',
                        transition={'duration':transition_duration},
                    )
                ]
        )
        slider_steps.append(step)
    sliders[0]['steps']=slider_steps


    #################### LAYOUT ##########################
    layout=go.Layout(
                title='Hit-Energy Map and Corresponding Saliency(Gradient) Map',
                xaxis1=dict(range=[0,input_img.shape[0]],
                            domain=[0,0.4],
                            mirror='ticks',
                            showline=True),
                yaxis1=dict(range=[0,input_img.shape[1]],
                            #domain=[0.6,1],
                            anchor='x1',
                            showline=True,
                            mirror='ticks'),
                xaxis2=dict(range=[0,input_img.shape[0]],
                            domain=[0.6,1]),
                yaxis2=dict(range=[0,input_img.shape[1]],
                            #domain=[0.6,1],
                            anchor='x2'),
                sliders=sliders,
    )
    fig['layout']=layout
    fig['frames']=img_grad_frames
    plotly.offline.plot(fig,filename=save_dir+'map.html')
