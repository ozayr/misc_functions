import PySimpleGUI as sg

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
# from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")
plt.style.use('dark_background')

def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def create_view(fig ,order = 311 ):
    ax = fig.add_subplot(order, projection='3d')
    ax.grid(False)
    ax.cla()
    ax.scatter(0, 0, 0, marker='^')
    ax.azim = 0
    ax.elev = 0
    return ax

def create_polar(fig,azim, title = '',order = 311):
    ax = fig.add_subplot(order, polar=True)
    theta = np.deg2rad([azim,azim])
    r = [0,1]
#     ax.plot(theta,r ,'r')
    ax.arrow(np.deg2rad(azim), 0, 0, 0.9, width = 0.015,
                 edgecolor = 'red', facecolor = 'red', lw = 3, zorder = 5)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(title)
    ax.set_rticks([])  # less radial ticks
    ax.yaxis.set_tick_params(labelsize=10)
    
    return ax

def manipulate_view(ax, pan = 0 ,tilt = 0 ,title = '', threat_loc = ''):
    ax.cla()
    ax.grid(False)
    ax.set_title(title)
   
    if threat_loc:
        x,y,z = threat_loc
        
        ax.scatter(x,y,z,s= 50, c = 'r',marker = 'o', label = 'Threat')
    ax.scatter(0, 0, 0, s = 25,c= 'b', marker='^' , label = 'vehicle')
    ax.legend()
    ax.plot([0,x],[0,y],[0,z],'y' )
    ax.autoscale(enable=False)  #you will need this line to change the Z-axis
    ax.set_xbound(-1, 1)
    ax.set_ybound(-1, 1)
    ax.set_zbound(0, 1)
    
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    
    ax.w_xaxis.pane.set_color('black')
    ax.w_xaxis.line.set_color('black')
    ax.w_yaxis.pane.set_color('black')
    ax.w_yaxis.line.set_color('black')
    
    ax.w_zaxis.pane.set_color('green')
    ax.w_zaxis.line.set_color('black')

    
    r = 0.9
    pi = np.pi
    cos = np.cos
    sin = np.sin
    # altitude
    phi, theta = np.mgrid[0.0:0.5*pi:20j, 0.0:2.0*pi:60j] # phi = alti, theta = azi
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)    
    
    ax.plot_surface(
    x, y, z,  rstride=4, cstride=4, color='b', alpha=0.4, linewidth=0)
    
    
    ax.azim = pan
    ax.elev = tilt
    

def show_threat(elevation = 0, azimuth = 0):
    
    
    elevation_rad = np.deg2rad(elevation)
    azimuth_rad = np.deg2rad(azimuth)
    x =  np.cos(elevation_rad) * np.cos(azimuth_rad)
    y =  np.cos(elevation_rad) * np.sin(azimuth_rad)
    z =  np.sin(elevation_rad)
    
    threat_location = [x, y , z]
    
    
    fig = plt.figure(figsize = (8,8))
    plt.rcParams['grid.color'] = "white"
    rotating_view = create_view(fig,order = 311)
    side_view = create_view(fig,order = 312)
    top_view = create_polar(fig,azimuth,title = f'Azimuth view: {azimuth}°' ,order= 313)
    
    

    figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds
    threat_graphic_layout = [
            [sg.Text('Threat Alert!',text_color = 'red',key = 'alert', justification='center', font='Helvetica 20')],
            [sg.Canvas(size=(figure_w, figure_h), key='-CANVAS-')],
            [sg.Button('Exit', size=(10, 1)) ]
    ]

    # create the form and show it without the plot
    threat_graphic_window = sg.Window('Incoming Threat',threat_graphic_layout, finalize=True)
    canvas_elem = threat_graphic_window['-CANVAS-']
    canvas = canvas_elem.TKCanvas

    # draw the initial plot in the window

    fig_agg = draw_figure(canvas, fig)

    fig_agg.draw()
    i = 0
    while 1:
        i = i+1
        event, values = threat_graphic_window.read(timeout=10)
        if event in ('Exit', None):
            break
        manipulate_view(rotating_view,pan = i ,tilt = 10,threat_loc = threat_location)
        manipulate_view(side_view,title = f'Elevation view: {elevation}°',pan = 0 ,tilt = 0,threat_loc = threat_location)
#         manipulate_view(top_view,title = 'azimuth view',tilt =-90,threat_loc = threat_location)
        plt.tight_layout()
    
    
        fig_agg.draw()

    threat_graphic_window.close()
	

show_threat(45,30)
