# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:29:15 2022

@author: ihipt
"""

import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as n
import numpy as np


class Shape:
    def __init__(self,p,cm,n,name=''):
        self.name = name # shape label e.g. lens
        self.p = p # All shape vertices in 3D (points)
        self.cm = cm # triangle connectivity matrix (sides)
        self.n = n # refractive index of inner medium
    def centre(self):
        '''determine centre point'''
        pass
    def rotate(self,axis):
        '''rotate shape around axis'''
    def scale(self):
        '''change size of shape by modifying points'''
        pass
    def translate(self,v):
        '''move shape in direction and distance of v'''
        self.p += v

class create_rays:
    def __init__(self,num,width,y,z,up,n):
        self.num = num # number of rays generated
        self.p = np.zeros((3,num)) # points
        self.p[0] = np.linspace(0,width,num) # x values
        self.p[1] = y*np.ones((1,num)) # y values
        self.p[2] = z*np.ones((1,num)) # z values
        self.up = np.tile(np.array(up)[...,None],(1,num)) # unit direction vectors
        self.pacc = np.copy(self.p) # accumulated p
        self.upacc = np.copy(self.up) # accumulated up
        self.dacc = np.zeros(num)# accumulated d
        self.origin = np.arange(num) # index of ray origin in dacc (mutable)
        self.n = np.repeat(n,num) # refractive index for current medium
        self.inshape = np.repeat(False,num) # within shape bool


def startupfcn():
    global numrays_slider_var
    global extend_rays_input
    numrays_slider_var.set(5)
    extend_rays_input.insert(tk.END,'1.5')
    width_input.insert(tk.END,'3.5')
    y_input.insert(tk.END,'-2')
    z_input.insert(tk.END,'1.2')
    upx.insert(tk.END,'0')
    upy.insert(tk.END,'1')
    upz.insert(tk.END,'0')
    createshape()
    trace()

def plotshapes():
    global shapes
    for shape in shapes:
        plot3d.scatter(shape.p[:,0],shape.p[:,1],shape.p[:,2])
        plot3d.set_xlim(-1,4)
        plot3d.set_ylim(-4,1)
        plot3d.set_zlim(-1,4)
        canvas.draw()

def createshape():
    global shapes
    shapes = [] # list of shape objects
    points = np.array([[1,-1,1],
                  [2,-1,1],
                  [1,-0,2],
                  [2,-1,2]])
    cm = np.array([[0,1,3],
                  [0,2,3]])
    shapes.append(Shape(points,cm,1.52))

def trace():
    global numrays_slider_var
    global shapes,rays
    
    '''Initialise rays'''
    rays = create_rays(int(numrays_slider_var.get()),
                       float(width_input.get()),
                       float(y_input.get()),
                       float(z_input.get()),
                       [float(upx.get()),float(upy.get()),float(upz.get())],#up
                       1) # refractive index of air
    
    '''Trace'''
    
    for ray in range(rays.num):
        cob = find_change_of_basis_matrix(rays.up[:,ray])
        ray_cob = np.matmul(cob,rays.p[:,ray])
        for shape in shapes:
            s_cob = np.zeros((3,len(shape.p))) # note change in dim order
            for i,point in enumerate(shape.p):
                s_cob[:,i] = np.matmul(cob,point)
            shortest_distance = np.inf
            for triangle in shape.cm:
                if triangle_interior(ray_cob[:2],
                                     s_cob[:2,triangle[0]],
                                     s_cob[:2,triangle[1]],
                                     s_cob[:2,triangle[2]]):
                    normal = plane_from_points(shape.p[triangle])
                    d = distance_line_plane(rays.p[:,ray],rays.up[:,ray],normal,shape.p[triangle[0]])
                    if d < shortest_distance:
                        POI = rays.p[:,ray]+d*rays.up[:,ray]
                        shortest_distance = d
                        closest_plane = normal
            if shortest_distance < np.inf:
                rays.p[:,ray] = POI
                rays.up[:,ray] = snells_law(rays.up[:,ray],closest_plane,rays.n[ray],shape.n,rays.inshape[ray])
                rays.pacc = np.column_stack((rays.pacc,rays.p[:,ray]))
                rays.upacc = np.column_stack((rays.upacc,rays.up[:,ray]))
                rays.dacc = np.append(rays.dacc,0)
                rays.dacc[rays.origin[ray]] = shortest_distance
                rays.origin[ray] = len(rays.dacc)-1
    extend_rays()
    plotrays()
            
    
def find_change_of_basis_matrix(v):
    # rearrange dot product formula
    orth1 = np.zeros((3,1))
    orth2 = np.zeros((3,1))
    dim = 0
    while v[(dim+2)%3] == 0:
        dim += 1
    orth1[(dim+0)%3] = 1
    orth1[(dim+1)%3] = 2
    orth1[(dim+2)%3] = -(orth1[(dim+0)%3]*v[(dim+0)%3]+orth1[(dim+1)%3]*v[(dim+1)%3])/v[(dim+2)%3]
    orth2[(dim+0)%3] = 1
    orth2[(dim+1)%3] = -2
    orth2[(dim+2)%3] = -(orth2[(dim+0)%3]*v[(dim+0)%3]+orth2[(dim+1)%3]*v[(dim+1)%3])/v[(dim+2)%3]
    P = np.concatenate((orth1,orth2,v[:].reshape(3,1)),axis=1)
    Pinv = np.linalg.inv(P)
    return Pinv

def triangle_interior(query,p1,p2,p3):
    v1 = p2-p1
    v2 = p3-p1
    a_ = (np.cross(query,v2)-np.cross(p1,v2))/np.cross(v1,v2)
    b_ = -(np.cross(query,v1)-np.cross(p1,v1))/np.cross(v1,v2)
    if a_>0:
        if b_>0:
            return(a_+b_<1)
        
def plane_from_points(points):
    # grab normal vector only
    v1 = points[2]-points[0]
    v2 = points[1]-points[0]
    return(np.cross(v1,v2))

def distance_line_plane(location,direction,n,ref):
    # loc vector and dir vector describe line
    # normal vector and reference point describe plane
    return np.dot(ref-location,-n)/np.dot(direction,-n)

def snells_law(line,plane,n1,n2,inshape):
    # line defined as direction vector
    # plane defined by normal vector
    # find new unit direction vector

    if inshape:
        r = n2/n1
    else:
        r = n1/n2
    c = np.dot(plane,line)
    if c<0:
        c = np.dot(-plane,line)
    v = r*line + (r*c-np.sqrt(1-r**2*(1-c**2)))*plane
    normv = v/np.linalg.norm(v) # unit vector must be normalised
    return normv

def extend_rays():
    global rays
    for i,d in enumerate(rays.dacc):
        if d==0:
            rays.dacc[i] = float(extend_rays_input.get())
            
def plotrays():
    global plot3d
    global rays
    upd = np.multiply(rays.upacc,rays.dacc)
    plot3d.quiver(rays.pacc[0],rays.pacc[1],rays.pacc[2],upd[0],upd[1],upd[2])
    plot3d.set_xlim(-1,4)
    plot3d.set_ylim(-4,1)
    plot3d.set_zlim(-1,4)
    plotshapes()
    canvas.draw()

def clear_button_callback():
    global plot3d
    plot3d.clear()
    plotshapes()
    canvas.draw()
        

'''setup code'''

window = tk.Tk()

window.title('3D Quiver Plot')

numrays_slider_var = tk.DoubleVar()



'''create widgets'''


trace_button = tk.Button(master = window,
                        command = trace,
                        text = "Trace")
clear_button = tk.Button(master=window,
                         comman=clear_button_callback,
                         text = "Clear")
numrays_slider = tk.Scale(master = window,
                          variable = numrays_slider_var,
                          from_ = 1,
                          to = 30,
                          orient = tk.HORIZONTAL)
extend_rays_input = tk.Entry(window)
extend_rays_label = tk.Label(window,text='Extend Rays')
width_input = tk.Entry(window)
width_label = tk.Label(window,text='width')
y_input = tk.Entry(window)
y_label = tk.Label(window,text='py')
z_input = tk.Entry(window)
z_label = tk.Label(window,text='pz')
upx = tk.Entry(window)
upx_label = tk.Label(window,text='upx')
upy = tk.Entry(window)
upy_label = tk.Label(window,text='upy')
upz = tk.Entry(window)
upz_label = tk.Label(window,text='upz')


'''Create Axes'''

fig = Figure(figsize = (5, 5), dpi = 100) # 500 x 500


canvas = FigureCanvasTkAgg(fig,master = window)
canvas.draw()
canvas.get_tk_widget().pack()
plot3d = fig.add_subplot(projection='3d')

startupfcn()

'''assemble window'''


numrays_slider.pack()
trace_button.pack()
clear_button.pack()
extend_rays_input.pack()
extend_rays_label.pack()
width_input.pack()
y_input.pack()
z_input.pack()
upx.pack()
upy.pack()
upz.pack()

clear_button.place(x=612-150,y=500)
trace_button.place(x=0,y=500)
numrays_slider.place(x=50,y=500)
extend_rays_label.place(x=312,y=500)
h = 19
width_label.place(x=312,y=500+h*1)
y_label.place(x=312,y=500+h*2)
z_label.place(x=312,y=500+h*3)
upx_label.place(x=312,y=500+h*4)
upy_label.place(x=312,y=500+h*5)
upz_label.place(x=312,y=500+h*6)
window.geometry('500x650')

window.mainloop()
