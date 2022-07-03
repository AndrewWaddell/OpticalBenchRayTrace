# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:29:15 2022

@author: ihipt
"""

import numpy as np


class Shape:
    def __init__(self,location,direction,n,name=''):
        self.name = name # shape label e.g. lens
        self.location = location # translation from source points
        self.direction # new orientation of x axis vector
        self.n = n # refractive index of inner medium


class Triangulated(Shape):
    def __init__(self,location,direction,n,p,cm,name=''):
        super().__init__(location,direction,n,name)
        self.p = p # All shape vertices in 3D (points)
        self.cm = cm # triangle connectivity matrix (sides)
    def rotate(self,axis_loc,axis_dir):
        '''rotate points clockwise around axis defined as a vector with location and direction'''
        pass
    def translate(self,v):
        '''move shape in direction and distance of v'''
        self.p += v
    def trace(self):
        pass

class Rays:
    def __init__(self,numrays,p,up,n):
        self.numrays = numrays # number of rays generated
        # self.p = np.zeros((num,3))
        # self.up = np.transpose(np.tile(np.array(up)[...,None],(1,num))) # unit direction vectors
        # for ray in range(numrays):
        #     theta = random.randint(0,1000)*2*np.pi/1000
        #     radius = random.randint(0,1000)/1000
        #     self.p[ray][0] = radius*np.cos(theta) + p[0] # x
        #     self.p[ray][1] = radius*np.sin(theta) + p[2] # y
        #     self.p[ray][2] = p[1] # z
        self.p = p
        self.up = up
        self.pacc = np.copy(self.p) # accumulated p
        self.upacc = np.copy(self.up) # accumulated up
        self.dacc = np.zeros(numrays)# accumulated d
        self.origin = np.arange(numrays) # index of ray origin in dacc (mutable)
        self.n = np.repeat(n,numrays) # refractive index for current medium
        self.inside = np.repeat(False,numrays) # within shape boolean (switches upon intersection)
        self.wavelength = np.repeat('visible',numrays) # visible or infrared - modify later to nm
        
class aspheric:
    def __init__(self,R,k,a4,a6,d):
        self.R = R
        self.k = k
        self.a4 = a4
        self.a6 = a6
        self.d = d
    def calculate(self,r):
        return self.d*(1/(self.R*(1+np.sqrt(1-(1+self.k)*(r**2/self.R**2)))) + self.a4*r**4 + self.a6*r**6 - 0.5)/0.5
    def points(self,depth):
        self.depth = depth
        self.x = [0]
        self.y = [0]
        self.z = [self.calculate(0)]
        for i in range(depth):
            radius = (i+1)/depth
            for i in range(depth):
                theta = i*2*np.pi/depth
                self.x.append(radius * np.cos(theta))
                self.y.append(radius * np.sin(theta))
                self.z.append(self.calculate(radius))
    def connectivity_matrix(self):
        self.cm = []
        '''make inner ring'''
        for i in range(self.depth-1):
            self.cm.append([0,i+1,i+2])
        self.cm.append([0,self.depth,1])
        '''make other rings'''
        for ring in range(self.depth-1):
            for i in range(self.depth-1):
                self.cm.append([self.depth*ring+1+i,self.depth*ring+2+i,self.depth*(ring+1)+2+i])
                self.cm.append([self.depth*ring+1+i,self.depth*(ring+1)+1+i,self.depth*(ring+1)+2+i])
            i = self.depth-1
            self.cm.append([self.depth*ring+1+i,self.depth*ring+1,self.depth*(ring+1)+1])
            self.cm.append([self.depth*ring+1+i,self.depth*(ring+1)+1+i,self.depth*(ring+1)+1])

def create_octahedron(shape):
    x = shape.p[:,0]
    y = shape.p[:,1]
    z = shape.p[:,2]
    p = []
    p.append([min(x), min(y), min(z)])
    p.append([max(x), min(y), min(z)])
    p.append([min(x), max(y), min(z)])
    p.append([max(x), max(y), min(z)])
    p.append([min(x), min(y), max(z)])
    p.append([max(x), min(y), max(z)])
    p.append([min(x), max(y), max(z)])
    p.append([max(x), max(y), max(z)])
    cm = [[0, 1, 3],
          [0, 2, 3],
          [0, 1, 5],
          [0, 4, 5],
          [2, 6, 7],
          [2, 3, 7],
          [1, 3, 7],
          [1, 5, 7],
          [0, 2, 6],
          [0, 4, 6],
          [4, 5, 7],
          [4, 6, 7]]
    p = np.asarray(p)
    cm = np.asarray(cm)
    return p, cm    


# Kept this function for understanding of how to index points
# def plotshapes():
#     global shapes
#     for shape in shapes:
#         '''lines'''
#         # decided not to show shape points
#         for triangle in shape.cm:
#             X = [shape.p[triangle[2]][0]]
#             Y = [shape.p[triangle[2]][1]]
#             Z = [shape.p[triangle[2]][2]]
#             for point in triangle:
#                 [x,y,z] = shape.p[point]
#                 X.append(x)
#                 Y.append(y)
#                 Z.append(z)
#             plot3d.plot(X,Y,Z)

def createshape():
    global shapes
    shapes = [] # list of shape objects
    lens1 = aspheric(1,0,0,0,0.3)
    lens1.points(10)
    pointsxyz= np.array([lens1.x,lens1.y,lens1.z])
    points = np.transpose(pointsxyz)
    lens1.connectivity_matrix()
    shapes.append(Shape(points,np.array(lens1.cm),1.52))

def trace():
    global numrays_slider_var
    global shapes,rays
    
    '''Initialise rays'''
    rays = create_rays(int(numrays_slider_var.get()),
                       [float(x_input.get()),float(y_input.get()),float(z_input.get())], #p
                       [float(upx.get()),float(upy.get()),float(upz.get())],#up
                       1) # refractive index of air
    '''Trace'''
    
    for ray in range(rays.num):
        cob = find_change_of_basis_matrix(rays.up[ray,:])
        ray_cob = np.matmul(cob,rays.p[ray,:])
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
                    d = distance_line_plane(rays.p[ray,:],rays.up[ray,:],normal,shape.p[triangle[0]])
                    if d < shortest_distance:
                        POI = rays.p[ray,:]+d*rays.up[ray,:]
                        shortest_distance = d
                        closest_plane = normal
            if shortest_distance < np.inf:
                rays.p[ray,:] = POI
                rays.up[ray,:] = snells_law(rays.up[ray,:],closest_plane,rays.n[ray],shape.n,rays.inshape[ray])
                rays.pacc = np.row_stack((rays.pacc,rays.p[ray,:]))
                rays.upacc = np.row_stack((rays.upacc,rays.up[ray,:]))
                rays.dacc = np.append(rays.dacc,0)
                rays.dacc[rays.origin[ray]] = shortest_distance
                rays.origin[ray] = len(rays.dacc)-1
    extend_rays()

    
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
    if all(p1 == p2) or all(p1 == p3) or all(p1 == p3): # if parallel to the triangle
        return 0
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
            
# Kept function to remind of the default data structure for rays
# def plotrays():
#     upd = np.multiply(np.transpose(rays.upacc),rays.dacc)
#     plot3d.quiver(x,y,z,upd[0],upd[1],upd[2])
        


