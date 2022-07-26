# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:29:15 2022

@author: ihipt
"""

import numpy as np
from scipy.io import savemat


class Scene:
    def __init__(self):
        self.sources = []
        self.rays = Rays()
        self.shapes = []
        self.n = 1 # refractive index of air
    def trace(self):
        for s in self.sources:
            self.rays.append(s.numrays,s.p,s.up)
        for s in self.shapes:
            s.trace(self)
    def add_source(self):
        self.sources.append(Source())
    def add_shape(self):
        # placeholder for now
        loc = np.array([0,0,0])
        direc = np.array([1,0,0])
        n = 1.52
        p = np.array([[-1,-1,1],
                      [3,3,2],
                      [-1,3,3],
                      [3,-1,4]])
        cm = np.array([[0,2,3],
                      [1,2,3]])
        self.shapes.append(Triangulated(loc,direc,n,p,cm,False,'Testglass'))

class Shape:
    def __init__(self,location,direction,n,mirror,name):
        self.name = name # shape label e.g. lens
        self.location = location # translation from source points
        self.direction = direction # new orientation of x axis vector
        self.n = n # refractive index of inner medium
        self.mirror = mirror


class Triangulated(Shape):
    def __init__(self,location,direction,n,p,cm,mirror,name):
        super().__init__(location,direction,n,mirror,name)
        self.p = p # All shape vertices in 3D (points)
        self.cm = cm # triangle connectivity matrix (sides)
    def rotate(self,axis_loc,axis_dir):
        '''rotate points clockwise around axis defined as a vector with location and direction'''
        pass
    def translate(self,v):
        '''move shape in direction and distance of v'''
        self.p += v
    def scale(self,k,v):
        '''Scale shape by scaling factors k outwards from location v. inputs are numpy arrays
        If uniform scaling, use k = [1,1,1]'''
        self.p += v
        self.p *= k
        self.p -= v
    def trace(self,scene):
        cob = change_of_basis(scene.rays.up)
        rays_cob = np.einsum('hij,hj->hi',cob,scene.rays.p) # perform change of basis to rays
        cobrep = np.repeat(cob[:,np.newaxis,:,:],self.p.shape[0],axis=1)  # broadcast in advance
        prep = np.repeat(self.p[np.newaxis,:,:],scene.rays.numrays,axis=0) # broadcast in advance
        shape_cob = np.einsum('ghij,ghj->ghi',cobrep,prep) # perform change of basis to shape
        # shape_cob format: [rays:points:dimensions=3]
        # shape for the following triangle points (before reshaping): [rays:triangles:dims=2]
        p1 = shape_cob[:,self.cm[:,0],:2] 
        p2 = shape_cob[:,self.cm[:,1],:2]
        p3 = shape_cob[:,self.cm[:,2],:2]
        p1r = p1.reshape(p1.shape[0]*p1.shape[1],2)
        p2r = p2.reshape(p2.shape[0]*p2.shape[1],2)
        p3r = p3.reshape(p3.shape[0]*p3.shape[1],2)
        rays_cob_rep = np.repeat(rays_cob[:,:2],self.cm.shape[0],axis=0) # broadcast rays over triangles
        interior = triange_interior(rays_cob_rep,p1r,p2r,p3r).reshape(scene.rays.numrays,self.cm.shape[0])
        cmrep = np.repeat(self.cm[np.newaxis,:,:],scene.rays.numrays,axis=0)
        triangle_points = self.p[cmrep[interior]]
        d = np.inf*np.ones(interior.shape)
        i = interior.nonzero()[0] # numerical index of intersecting rays
        normals = plane_from_points(triangle_points)
        d[interior.nonzero()] = distance_line_plane(scene.rays.p[i],
                            scene.rays.up[i],normals,triangle_points[:,0])
        min_distances = np.min(d,axis=1)
        scene.rays.p[i] = scene.rays.p[i] + min_distances[i] * scene.rays.up[i]
        if self.mirror:
            scene.rays.up[i] = reflect(scene.rays.up[i],normals)
        else:
            scene.rays.up[i] = refract(scene.rays.inside[i],scene.n,self.n,scene.rays.up[i],normals)
            scene.rays.inside[i] = np.logical_not(scene.rays.inside[i])
        scene.rays.pacc = np.concatenate((scene.rays.pacc,scene.rays.p[i]),axis=0)
        scene.rays.upacc = np.concatenate((scene.rays.upacc,scene.rays.up[i]),axis=0)
        scene.rays.dacc[i] = min_distances[i]
        scene.rays.dacc = np.concatenate((scene.rays.dacc,np.zeros(i.shape)),axis=0)

class Source:
    def __init__(self):
        # placeholder values for now
        self.numrays = 9
        x = np.repeat(np.arange(3),3)
        y = np.tile(np.arange(3),3)
        z = np.zeros(9)
        self.p = np.column_stack((x,y,z))
        x = np.zeros(9)
        y = np.zeros(9)
        z = np.ones(9)
        self.up = np.column_stack((x,y,z))

class Rays:
    def __init__(self):
        self.numrays = 0 # number of rays generated
        self.p = np.array([]) # coordinate vectors
        self.up = np.array([]) # unit vectors
        self.pacc = np.array([]) # accumulated p
        self.upacc = np.array([]) # accumulated up
        self.dacc = np.array([]) # accumulated d
        self.origin = np.array([]) # index of ray origin in dacc (mutable)
        self.inside = np.array([]) # within shape boolean (switches upon intersection)
        self.wavelength = np.array([]) # visible or infrared - modify later to nm
    def append(self,numrays,p,up,wavelength='visible'):
        self.numrays += numrays
        self.p = np.concatenate((self.p,p)) if self.p.size else np.copy(p)
        self.up = np.concatenate((self.up,up)) if self.up.size else np.copy(up)
        self.pacc = np.concatenate((self.pacc,p)) if self.pacc.size else np.copy(p)
        self.upacc = np.concatenate((self.upacc,up)) if self.upacc.size else np.copy(up)
        dacc = np.zeros(numrays)
        self.dacc = np.concatenate((self.dacc,dacc)) if self.dacc.size else dacc
        start = np.max(self.origin)+1 if self.origin.size else 0
        origin = np.arange(start,start+numrays)
        self.origin = np.concatenate(self.origin,origin) if self.origin.size else origin
        inside = np.repeat(False,numrays)
        self.inside = np.concatenate((self.inside,inside)) if self.inside.size else inside
        wavelength = np.repeat(wavelength,numrays)
        self.wavelength = np.concatenate((self.wavelength,wavelength)) if self.wavelength.size else wavelength
        
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

def createshape():
    global shapes
    shapes = [] # list of shape objects
    lens1 = aspheric(1,0,0,0,0.3)
    lens1.points(10)
    pointsxyz= np.array([lens1.x,lens1.y,lens1.z])
    points = np.transpose(pointsxyz)
    lens1.connectivity_matrix()
    shapes.append(Shape(points,np.array(lens1.cm),1.52))

def change_of_basis(v):
    '''computes a non-unique change of basis matrix for a vector in R3.
    The 3rd dimension of the basis set is along the direction of v.
    This function operates on multiple vectors at once where v is a 
    matrix consisting of each vector nested within it. v is shape
    (numrays,3)'''
    numrays = v.shape[0]
    orth1 = rotate_3d_vector_90(v)
    orth2 = np.cross(orth1,v)
    P = np.concatenate((orth1,orth2,v),axis=1).reshape(numrays,3,3).transpose(0,2,1)
    Pnorm = np.divide(P,np.tile(np.linalg.norm(P,axis=1),3).reshape(numrays,3,3))
    Pinv = np.linalg.inv(Pnorm)
    return Pinv

def rotate_3d_vector_90(v):
    '''Creates 3 rotation matrices evaluated at theta = pi/2.
    Rotates the input vector by 90deg over each axis, one at
    a time to give an orthogonal vector in R3'''
    # Create rotation matrices
    Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    Ry = np.array([[0,0,1],[0,1,0],[-1,0,0]])
    Rz = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    # Broadcast matrices to accommodate several input vectors
    Rxr = np.repeat(Rx[np.newaxis,:,:],v.shape[0],axis=0)
    Ryr = np.repeat(Ry[np.newaxis,:,:],v.shape[0],axis=0)
    Rzr = np.repeat(Rz[np.newaxis,:,:],v.shape[0],axis=0)
    # Rotate vectors by matrix multiplication
    v_rx = np.einsum('ijk,ik->ij',Rxr,v)
    v_rx_ry = np.einsum('ijk,ik->ij',Ryr,v_rx)
    v_rx_ry_rz = np.einsum('ijk,ik->ij',Rzr,v_rx_ry)
    return v_rx_ry_rz

def GramSmchidt(u1):
    '''produces one new vector u2 that is orthogonal to vector u1.
    u1 is in R3 and must have 3 entries.
    u1 is vectorised to work on multiple rays and should be shape
    (numrays,3)'''
    v = np.copy(u1) # first create an arbitrary vector v, that isn't u
    while np.any(np.all(v==u1,axis=1)): # for the unlikely event any random v gives u
        v = np.random.randn(u1.shape[0],3)
    u2 = v - projection_operator(u1,v)
    return u2

def projection_operator(u,v):
    '''projects the vector v orthogonally onto the line spanned by vector u
    inputs u and v are vectorised into a matrix where each nested entry 
    represents each ray'''
    numrays = u.shape[0]
    udotv = np.inner(u,v)[np.nonzero(np.eye(numrays))]
    udotu = np.square(np.linalg.norm(u,axis=1))
    scalar = np.divide(udotv,udotu)
    proj_u_v = np.multiply(u,np.repeat(scalar,3).reshape(numrays,3))
    return proj_u_v

def triange_interior(query,p1,p2,p3):
    ''' Tests whether the query point fits within the triangle created by
    points p1, p2 and p3. However, this function operates on multiple
    triangles at once, and multiple rays at once. Luckily, the input
    arrays have been reduced to a single list of rays and triangles.
    This means the input shape for each argument is testcase*3dims.
    For example, 2 triangles and 4 rays will give us shape = 8*3
    
    Algorithm checks the number of points in the convex hull. If 4,
    point is outside triangle, if 3 then point is inside.
    
    
    
    Does it deal with case of being parallel to triange?'''
    # Solve query = p1 + a*(p2-p1) + b*(p3-p1)
    # query - p1 = [p2:p3]-p1 * [a_b]
    M = np.concatenate((p2[:,np.newaxis,:],p3[:,np.newaxis,:]),axis=1) # M = p2:p3
    p1r = np.repeat(p1,2,axis=0).reshape(p1.shape[0],2,2) # broadcast p1 into M
    a_b = np.linalg.solve(M-p1r,query-p1)
    a_and_b_gt_0 = np.all(np.greater(a_b,np.zeros((a_b.shape[0],2))),axis=1)
    a_plus_b_lt_1 = np.greater(np.ones(18),np.sum(a_b,axis=1))
    return np.bitwise_and(a_and_b_gt_0,a_plus_b_lt_1)

def plane_from_points(p):
    '''Finds the normal vector orthogonal to the plane described by 3 points
    in a triangle. Function works for n triangles, Input points p are shape:
    n*3dims
    Note, we don't need to normalise the normals because distance line-plane
    equation will work anyway'''
    v1 = p[:,2] - p[:,0]
    v2 = p[:,1] - p[:,0]
    n = np.cross(v1,v2)
    return n/np.linalg.norm(n,axis=1)

def distance_line_plane(l0,l,n,p0):
    '''For the set of points (p) on a line described by p = l0 + l*d, where:
    l0 is a point on the line,
    l is a unit vector in the direction of the line,
    d is a scalar;
    and for the set of points (p) on a plane described by
    dot( (p - p0) , n ) = 0, where:
    n is a normal vector to the plane,
    p0 is a point on the plane;
    substituting p line into p plane, and re-arranging to solve for d
    will give us the distance from l0 to the plane, since l is normalised.
    
    Function works for multiple line,plane pairs, which are orded in the
    first dimension of the input array.
    
    Plane normals must be normalised.
    '''
    numerator = np.einsum('ij,ij->i',p0-l0,n) # dot product
    denominator = np.einsum('ij,ij->i',l,n) # dot product
    d = np.divide(numerator,denominator)
    return d

def refract(inside,n1,n2,line,plane):
    '''Implements Snell's law in vector form for multiple line-plane pairs.
    n1 is refractive index of the atmosphere of the scene,
    n2 is refractive index of the shape.
    Therefore we cannot have shape-shape intersections as of yet.
    Ray must enter shape from atmosphere, then exit back into atmosphere.
    However, some rays may be entering the shape while others are exiting
    it, this is made possible because a mirror has an odd number of
    intersections while glass has an even number.
    Refracted ray is v.
    '''  
    r = np.zeros(inside.shape) # refractive index ratio for each ray
    r[inside] = n2/n1 # for rays that are exiting
    r[np.logical_not(inside)] = n1/n2 # for rays that are entering
    c1,plane = cos_theta1(plane,line) # note: updates normal vectors
    c2 = np.sqrt(1-np.multiply(np.square(r),1-np.square(c1))) # cos(theta2)
    rc1mc2 = np.multiply(r,c1) - c2 # r*cos(theta1) minus cos(theta2)
    rl = np.multiply(np.repeat(r[:,np.newaxis],3,axis=1),line)
    v = rl + np.multiply(np.repeat(rc1mc2[:,np.newaxis],3,axis=1),plane)
    return v

def reflect(line,plane): # mirror
    ''' Implements Snell's law for multiple line plane pairs.
    If normal vector doesn't point towards direction of the incoming ray,
    then cos(theta1) would be negative, so we negate n to fix this.
    '''
    c1,plane = cos_theta1(plane,line)
    return line + 2*np.multiply(np.repeat([c1],3,axis=1).reshape(c1.shape[0],3),plane)

def cos_theta1(plane,line):
    '''If normal vector doesn't point towards direction of the incoming ray,
    then cos(theta1) would be negative, so we negate n to fix this.
    Implemented within a Snells Law function'''
    plane = np.divide(plane,np.linalg.norm(plane,axis=1)) # normalise vectors
    c1_with_negs = np.einsum('ij,ij->i',-plane,line) # dot product
    plane[c1_with_negs<0] *= -1
    c1 = np.einsum('ij,ij->i',-plane,line) # repeat with correct normals
    return c1,plane



test_bench = Scene()
test_bench.add_source()
test_bench.add_shape()
test_bench.trace()


mat_workspace = {"pacc":test_bench.rays.pacc,
                 "upacc":test_bench.rays.upacc,
                 "dacc":test_bench.rays.dacc,
                 "sp":test_bench.shapes[0].p,
                 "cm":test_bench.shapes[0].cm
    }
savemat("python_outputs.mat", mat_workspace)
