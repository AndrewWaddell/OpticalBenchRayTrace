# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:29:15 2022

@author: ihipt
"""

import numpy as np
from scipy.io import savemat
from scipy.spatial import ConvexHull


class Scene:
    def __init__(self):
        self.sources = []
        self.rays = Rays()
        self.shapes = []
        self.n = 1 # refractive index of air
    def trace(self):
        for s in self.sources:
            self.rays.append(s.numrays,s.p,s.up)
        self.rays.change_of_basis()
        d = np.inf * np.ones((len(self.shapes),self.rays.numrays))
        for i,s in enumerate(self.shapes):
            s.change_of_basis(self) # grab shape points in terms of rays bases
            if s.trace_low_res(self): # is shape in line of sight? (LoS)
                d[i,:] = s.trace_d(self) # find distance to shape for each ray
        # choose shape with lowest d for each ray
        # update p & up for each selected shape intersection - using s.trace_save(d)
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
    def trace_low_res(self,scene):
        '''Determines if any of the rays intersect anywhere within this shape
        We iterate over all of the rays one at a time.
        First we construct a convex hull for the shape alone, and determine the number of points
        that form the hull (number of vertices), num_v.
        Then we concatenate the hull vertices with the ray and calculate a new hull, we call our test.
        If there are more points in the hull, then the point lies outside the shape, and does not intersect.
        In this case, we just continue to check for the next ray. If any ray intersects the shape then we
        return a successful result.
        If none of the rays intersect with the shape we return False.
        '''
        for i,shape_p in enumerate(self.p_cob): # for each ray
            shape_hull = ConvexHull(shape_p[:,:2])
            num_v = shape_hull.vertices.shape[0]
            test_case = np.concatenate((shape_hull.points[shape_hull.vertices],scene.rays.rays_cob[np.newaxis,i,:2]))
            hull_test = ConvexHull(test_case)
            if hull_test.vertices.shape[0] <= num_v:
                return True
        return False
    def change_of_basis(self,scene):
        '''Calculates shape points in terms of new basis defined by ray directions'''
        cobrep = np.repeat(scene.rays.cob[:,np.newaxis,:,:],self.p.shape[0],axis=1) # broadcast in advance
        prep = np.repeat(self.p[np.newaxis,:,:],scene.rays.numrays,axis=0) # broadcast in advance
        self.p_cob = np.einsum('ghij,ghj->ghi',cobrep,prep) # perform change of basis to shape
    def trace_d(self,scene):
        '''Find the distance to the shape for each ray intersection'''
        # p_cob format: [rays:points:dimensions=3]
        # shape for the following triangle points (before reshaping): [rays:triangles:dims=2]
        p1 = self.p_cob[:,self.cm[:,0],:2] 
        p2 = self.p_cob[:,self.cm[:,1],:2]
        p3 = self.p_cob[:,self.cm[:,2],:2]
        p1r = p1.reshape(p1.shape[0]*p1.shape[1],2)
        p2r = p2.reshape(p2.shape[0]*p2.shape[1],2)
        p3r = p3.reshape(p3.shape[0]*p3.shape[1],2)
        rays_cob_rep = np.repeat(scene.rays.rays_cob[:,:2],self.cm.shape[0],axis=0) # broadcast rays over triangles
        interior = triangle_interior(rays_cob_rep,p1r,p2r,p3r).reshape(scene.rays.numrays,self.cm.shape[0])
        cmrep = np.repeat(self.cm[np.newaxis,:,:],scene.rays.numrays,axis=0)
        triangle_points = self.p[cmrep[interior]]
        d = np.inf*np.ones(interior.shape)
        self.i = interior.nonzero()[0] # numerical index of intersecting rays
        self.normals = plane_from_points(triangle_points)
        d[interior.nonzero()] = distance_line_plane(scene.rays.p[self.i],
                            scene.rays.up[self.i],self.normals,triangle_points[:,0])
        self.min_distances = np.min(d,axis=1)
        return self.min_distances
    def trace_save(self,scene):
        broadcasted_d = np.repeat(self.min_distances[self.i,np.newaxis],3,axis=1)
        scene.rays.p[self.i] = scene.rays.p[self.i] + broadcasted_d * scene.rays.up[self.i]
        if self.mirror:
            scene.rays.up[self.i] = reflect(scene.rays.up[self.i],self.normals)
        else:
            scene.rays.up[self.i] = refract(
                scene.rays.inside[self.i],
                scene.n,self.n,
                scene.rays.up[self.i],
                self.normals)
            scene.rays.inside[self.i] = np.logical_not(scene.rays.inside[self.i])
        scene.rays.pacc = np.concatenate((scene.rays.pacc,scene.rays.p[self.i]),axis=0)
        scene.rays.upacc = np.concatenate((scene.rays.upacc,scene.rays.up[self.i]),axis=0)
        scene.rays.dacc[self.i] = self.min_distances[self.i]
        scene.rays.dacc = np.concatenate((scene.rays.dacc,np.zeros(self.i.shape)),axis=0)

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
    def change_of_basis(self):
        '''computes a non-unique change of basis matrix for a vector in R3.
        The 3rd dimension of the basis set is along the direction of up.
        This function operates on multiple vectors at once where up is a 
        matrix consisting of each vector nested within it. up is shape
        (numrays,3)'''
        orth1 = rotate_3d_vector_90(self.up)
        orth2 = np.cross(orth1,self.up)
        P = np.concatenate((orth1,orth2,self.up),axis=1).reshape(self.numrays,3,3).transpose(0,2,1)
        Pnorm = np.divide(P,np.tile(np.linalg.norm(P,axis=1),3).reshape(self.numrays,3,3))
        self.cob = np.linalg.inv(Pnorm)
        self.rays_cob = np.einsum('hij,hj->hi',self.cob,self.p) # perform change of basis to rays
        
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

def triangle_interior(query,p1,p2,p3):
    ''' Tests whether the query point fits within the triangle created by
    points p1, p2 and p3. However, this function operates on multiple
    triangles at once, and multiple rays at once. Luckily, the input
    arrays have been reduced to a single list of rays and triangles.
    This means the input shape for each argument is testcases*3dims.
    For example, 2 triangles and 4 rays will give us shape = 8*3
    
    Algorithm checks the number of points in the convex hull. If 4,
    point is outside triangle, if 3 then point is inside or on edge.'''
    interior = np.zeros(query.shape[0])==np.zeros(query.shape[0])
    for i,q in enumerate(query):
        p = np.concatenate((q,p1[i],p2[i],p3[i]),axis=0).reshape(4,2)
        hull = ConvexHull(p)
        if hull.vertices.shape[0]==4:
            interior[i] = False
    return interior

def plane_from_points(p):
    '''Finds the normal vector orthogonal to the plane described by 3 points
    in a triangle. Function works for n triangles, Input points p are shape:
    n*3points*3dims'''
    v1 = p[:,2,:] - p[:,0,:]
    v2 = p[:,1,:] - p[:,0,:]
    n = np.cross(v1,v2)
    l = np.linalg.norm(n,axis=1) # length of each normal vector
    lr = np.repeat(l,3)
    return n/lr.reshape(n.shape)

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
    l = np.linalg.norm(plane,axis=1) # vector lengths
    lr = np.repeat(l,3).reshape(plane.shape) # broadcast
    plane = np.divide(plane,lr) # normalise vectors
    c1_with_negs = np.einsum('ij,ij->i',-plane,line) # dot product
    plane[c1_with_negs<0] *= -1
    c1 = np.einsum('ij,ij->i',-plane,line) # repeat with correct normals
    return c1,plane



test_bench = Scene()
test_bench.add_source()
test_bench.add_shape()
test_bench.trace()

# output to matlab for plotting:

mat_workspace = {"pacc":test_bench.rays.pacc,
                 "upacc":test_bench.rays.upacc,
                 "dacc":test_bench.rays.dacc,
                 "sp":test_bench.shapes[0].p,
                 "cm":test_bench.shapes[0].cm
    }
savemat("python_outputs.mat", mat_workspace)
