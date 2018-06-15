import plotly
import plotly.graph_objs as go
import numpy as np

####################### COLORSCALES ######################
def get_colorscale_view_hits():
    '''
    DESCRIPTION:
        This color scale is best fot viewing all the hits
        regardless of their energy. We are not stressing on the
        color of hits to be proportional to their energy.
    USAGE:
        OUTPUT:
            colorscale  : the color scale list ready to plug
    '''
    colorscale=[
                [0,'rgb(255,255,255)'],
                [0.01,'rgb(10,10,10)'],
                [1,'rgb(0.0,0.0,0.0)']
                ]
    return colorscale

def get_colorscale_view_energy():
    '''
    DESCRIPTION:
        In this color scale we will map the color proportional to
        their energy.
        Currently we will be supporting only the black-white
        variation.
    '''
    colorscale=[
                [0,'rgb(255,255,255)'],
                [1,'rgb(0.0,0.0,0.0)']
                ]
    return colorscale

####################### DATA ATTRIBUTES ##################
#Configuring the coorbars for the ticks
colorbar=dict(
            # tickmode='array',
            # tickvals=np.linspace(-1,1,10),
            # ticktext=np.linspace(-1,1,10),
            # tick0=-10,
            # dtick=2
)

####################### LAYOUT ATTRIBUTES ################
#Creating sliders and setting its properties
sliders=[{
    'active':0,
    #Positioning of the sliders
    'yanchor':'top',
    'xanchor':'left',
    'x':0.1,
    'y':0,
    #Padding and length of the sliders
    'pad':{'t':50,'b':10},
    'len':0.9,
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

#Creating the buttons for animation control
updatemenus=[{
    'type':'buttons',
    'buttons':[{#Button1
                'label':'Play',
                'method':'animate',
                'args':[None,
                        dict(frame=dict(duration=300),
                            transition=dict(duration=100,
                                        easing='quadratic-in-out'),
                            fromcurrent=True,
                            mode='immediate')
                        ]
                },
                {#Button2
                'label':'Pause',
                'method':'animate',
                'args':[[None],
                        dict(frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0),
                            mode='immediate')
                        ]
                }
    ],
    #Padding and positioning of buttons
    'pad': {'r': 10, 't': 87},
    'x':0.1,
    'y':0,
    'direction': 'left',
    'xanchor':'right',
    'yanchor':'top'
}]

margin=go.Margin(
    pad=50
)

def get_scene(dimension):
    '''
    DESCRIPTION:
        This function will set up the axis for the domain of data
        and setting the properties of the axis planes like
        background properties,color,range etc.

        Also, we will set the camera view-point here which will
        enable us to set the vantage point to see out event.
    USAGE:
        INPUT:
            dimension   : the dim in format [h,w,c] format
        OUTPUT:
            scene_dict  : the dictionary containing th configuration
                            of current scene properties.
    '''
    #Setting up the vantage-point
    '''
        the original vantage point is greate for 3d visualization,
        and color is good from that vantage point.
        But we can always reach that default camera position by
        pressing the home button.
    '''
    camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=-0.1, z=-1.75)
            )

    #Now describing the features of the scene (background)
    nx,ny,layers=dimension
    scene=dict(
            xaxis=dict(range=[0, nx],
                        autorange=False,
                        backgroundcolor="#e6ffff",
                        gridcolor="#003300",
                        showbackground=True,
                        zerolinecolor="#003300"
                    ),
            yaxis=dict(range=[0, ny],
                        autorange=False,
                        backgroundcolor="#e6ffff",
                        gridcolor="#003300",
                        showbackground=True,
                        zerolinecolor="#003300"
                        ),
            zaxis=dict(range=[0-2,layers+2],
                        autorange=False,
                        backgroundcolor="#e6ffff",
                        gridcolor="#003300",
                        showbackground=True,
                        zerolinecolor="#003300"
                        ),
            camera=camera,
            )

    return scene
