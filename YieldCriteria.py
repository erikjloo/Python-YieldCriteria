import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch

def calc_VonMises(x,y,z):
    return np.sqrt(0.5*((x-y)**2+(y-z)**2+(z-x)**2)) - 1

def calc_MohrCoulomb(s,phi,c):
    x,y,z,f = s[0],s[1],s[2],np.zeros(6)
    f[0] = 1/2*(x-y) + 1/2*(x+y)*np.sin(phi) - c*np.cos(phi)
    f[1] = 1/2*(y-x) + 1/2*(y+x)*np.sin(phi) - c*np.cos(phi)
    f[2] = 1/2*(y-z) + 1/2*(y+z)*np.sin(phi) - c*np.cos(phi)
    f[3] = 1/2*(z-y) + 1/2*(z+y)*np.sin(phi) - c*np.cos(phi)
    f[4] = 1/2*(x-z) + 1/2*(x+z)*np.sin(phi) - c*np.cos(phi)
    f[5] = 1/2*(z-x) + 1/2*(z+x)*np.sin(phi) - c*np.cos(phi)
    return max(f)

#=================================================================
# Orthogonal_proj
#=================================================================

def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    # -0.0001 added for numerical stability as suggested in:
    # http://stackoverflow.com/questions/23840756
    return np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,a,b],
                    [0,0,-0.0001,zback]])

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
# Later in your plotting code ...
proj3d.persp_transformation = orthogonal_proj

def orthogonal_plot():
    fig = plt.figure(figsize=(6,6))
    return fig.add_subplot(111, projection='3d')

def transformation():
    x3 = [1,1,1]/np.sqrt(3)
    x2 = [-1,-1,2]/np.sqrt(6)
    x1 = np.cross(x2,x3)
    return np.transpose(np.array([x1,x2,x3]))

def format_pi_plot(ax,radius,**kwargs):

    axes = kwargs.get('axes')
    scale = 15
    if axes == 'pos':
        x = Arrow3D([0,1.5*radius],[0,0],[0,0],mutation_scale = scale,lw=1.5, arrowstyle="-|>", color="k")
        y = Arrow3D([0,0],[0,1.5*radius],[0,0],mutation_scale = scale,lw=1.5, arrowstyle="-|>", color="k")
        z = Arrow3D([0,0],[0,0],[0,1.5*radius],mutation_scale = scale,lw=1.5, arrowstyle="-|>", color="k")
        ax.add_artist(x);ax.text(1.6*radius,0,0,'$\sigma_1$')
        ax.add_artist(y);ax.text(0,1.5*radius,0,'$\sigma_2$')
        ax.add_artist(z);ax.text(0,0,1.5*radius,'$\sigma_3$')
        ax.set_xlim(0, 1.4*radius)
        ax.set_ylim(0, 1.4*radius)
        ax.set_zlim(0, 1.4*radius)
    elif axes == 'neg':
        a = Arrow3D([0,-1.2*radius],[0,0],[0,0],mutation_scale = scale,lw=1.5, arrowstyle="-|>", color="k")
        b = Arrow3D([0,0],[0,-1.2*radius],[0,0],mutation_scale = scale,lw=1.5, arrowstyle="-|>", color="k")
        c = Arrow3D([0,0],[0,0],[0,-1.2*radius],mutation_scale = scale,lw=1.5, arrowstyle="-|>", color="k")
        ax.add_artist(a);ax.text(-1.3*radius,0,0,'$-\sigma_1$')
        ax.add_artist(b);ax.text(0,-1.4*radius,0,'$-\sigma_2$')
        ax.add_artist(c);ax.text(0,0,-1.2*radius,'$-\sigma_3$')
        ax.set_xlim(-1*radius,0.2*radius)
        ax.set_ylim(-1*radius,0.2*radius)
        ax.set_zlim(-1*radius,0.2*radius)
    else:
        print("Please set axes to pos or neg")
        
    # Format labels
    font_path = r'C:\Windows\Fonts\times.ttf'
    font_prop = font_manager.FontProperties(fname = font_path, size = 11)
    ax.get_xaxis().set_visible(False)
    ax.axis('off')

    # Set view
    angle = np.arctan(1/np.sqrt(2))*180/np.pi
    ax.view_init(angle,45)

#=================================================================
#   Mises_Tresca
#=================================================================

def plot_Mises_Tresca(ax,sig,fy):
    """ Plots Von Mises and Tresca on the pi-plane
        Input: sig = a stress vector (plots a point)
               fy = yield stress """
    
    # Tresca yield criterion
    t1 = np.array([fy,fy,0,0,0,fy,fy])
    t2 = np.array([0,fy,fy,fy,0,0,0])
    t3 = np.array([0,0,0,fy,fy,fy,0])

    # Circle (Von Mises)
    th = np.linspace(0,2*np.pi,100)
    vm = np.zeros([3,len(th)])
    radius = np.sqrt(2)*fy/2/np.cos(30*np.pi/180)
    vm[0,:] = np.cos(th)*radius
    vm[1,:] = np.sin(th)*radius
    vm[2,:] = np.ones(len(th))*0.6*fy

    # Transformation matrix
    Gam = transformation()
    
    # Von Mises yield criterion
    vm = np.dot(Gam,vm)

    # Plot
    if ax is None:
        ax = orthogonal_plot()
        ax.plot(t1,t2,t3,'r-')
        ax.plot(vm[0,:],vm[1,:],vm[2,:],'r-')
        format_pi_plot(ax,fy,axes = 'pos')
    
    # Plot Points
    ax.plot( [sig[0]], [sig[1]], [sig[2]], marker='o', markersize=6, color="blue")
    tx = ('$\sigma_1$ = {:1.0f} MPa').format(sig[0]); ax.text(sig[0],sig[1],sig[2],tx)

    return ax

#=================================================================
#   DruckerPrager
#=================================================================

def plot_DruckerPrager(ax,c,phi,fy):
    """ Plots Drucker Prager on 3D
        Input: c = coefficient of ??
               phi = angle of ??
               fy = radius of the pi-plane """
    
    # Cone (Drucker-Prager)
    MaxTensileStress = c*np.cos(phi)/np.sin(phi)
    th = np.linspace(0,2*np.pi,30)
    v = np.linspace(-1.4*MaxTensileStress,MaxTensileStress,10)
    [th,v] = np.meshgrid(th,v)
    X = (v-MaxTensileStress)*np.cos(th)*c
    Y = (v-MaxTensileStress)*np.sin(th)*c
    Z = (v)

    # Transformation matrix
    Gam = transformation()
    
    # Drucker-Prager yield criterion
    for r in range(np.size(X,0)):
        XYZ = [X[r,:],Y[r,:],Z[r,:]]
        xyz = np.dot(Gam,XYZ)
        X[r,:] = xyz[0,:]
        Y[r,:] = xyz[1,:]
        Z[r,:] = xyz[2,:]

    # Plot
    if ax is None:
        ax = orthogonal_plot()
        surf = ax.plot_surface(X, Y, Z, cmap=cm.Purples, linewidth=0, antialiased=False, alpha = 0.3)
        format_pi_plot(ax,fy,axes = 'neg')
        
    # Set view
    angle = np.arctan(1/np.sqrt(2))*180/np.pi
    ax.view_init(190,100)
    ax.view_init(180+angle,45)

    return ax


#=================================================================
#   VonMises
#=================================================================

def plot_VonMises(ax,fy):
    """ Plots Von Mises on 3D
        Input: fy = yield stress """
    
    # Cylinder (Von-Mises)
    th = np.linspace(0,2*np.pi,30)
    v = np.linspace(-fy/2,fy,10)
    [th,v] = np.meshgrid(th,v)
    X = np.cos(th)*fy/3
    Y = np.sin(th)*fy/3
    Z = (v)
    
    # Transformation matrix
    Gam = transformation()

    # Drucker-Prager yield criterion
    for r in range(np.size(X,0)):
        XYZ = [X[r,:],Y[r,:],Z[r,:]]
        xyz = np.dot(Gam,XYZ)
        X[r,:] = xyz[0,:]
        Y[r,:] = xyz[1,:]
        Z[r,:] = xyz[2,:]

    # Plot
    if ax is None:
        ax = orthogonal_plot()
        surf = ax.plot_surface(X, Y, Z, cmap=cm.Purples, linewidth=0, antialiased=False, alpha = 0.3)
        format_pi_plot(ax,fy,axes = 'pos')
        
    # Set view
    ax.view_init(10,10)

    return ax
    
#=================================================================
#   MohrCoulomb
#=================================================================

def plot_MohrCoulomb (ax,sig,phi,c):
    if ax is None:
        ax = orthogonal_plot()
        format_pi_plot(ax,2*c,axes = 'pos')

    # Pi-plane equation (assuming z = 0)
    from sympy.abc import x,y
    x0,y0,z0 = sig[0],sig[1],sig[2];
    plane = (x-x0) + (y-y0) - z0
    
    # Values to build normal to yield surface
    na = 1/2*(1+np.sin(phi))
    nb = -1/2*(1-np.sin(phi))
    

    # Normal to yield surface
    n = np.array([[na,nb,0],[na,0,nb],[0,na,nb],
                  [nb,na,0],[nb,0,na],[0,nb,na]])

    # Normal to Pi-plane
    p = np.array([1,1,1])

    s1 = np.zeros((7,3))
    s0 = np.zeros((6,3))
    ds = np.zeros((6,3))
    
    # Solve for 6 intersection lines in Pi-plane
    for i in range(6):

        # Intersection between plane and yield function
        ds[i] = np.cross(p,n[i])
        
        # Solve for s0 by assuming s0[2] = sig_3 = 0
        fi = n[i,0]*x + n[i,1]*y - c*np.cos(phi)
        sol = sp.solve([fi, plane])
        s0[i,0] = sol[x]; s0[i,1] = sol[y];
        
        # Update s0 so it is closest to the origin
        s0[i] += -np.dot(s0[i],ds[i])/np.dot(ds[i],ds[i])*ds[i]

        # Plot point s0[i] and arros n[i]
        ax.plot( [s0[i,0]], [s0[i,1]], [s0[i,2]], marker='o', markersize=3, color="blue")
        a = Arrow3D([s0[i,0],s0[i,0]+n[i,0]],[s0[i,1],s0[i,1]+n[i,1]],[s0[i,2],s0[i,2]+n[i,2]],
                 mutation_scale = 10,lw= 1.2, arrowstyle="->", color="b")
        ax.add_artist(a)
        tx = ('$n_{:1.0f}$').format(i+1); ax.text(s0[i,0],s0[i,1],s0[i,2],tx)

    for i in range(5):

        # Intersection between plane and adj. yield functions
        eqns = s0[i,0:2] + x*ds[i,0:2] - s0[i+1,0:2] + y*ds[i+1,0:2]
        sol = sp.solve(eqns)
        s1[i] = s0[i] + sol[x]*ds[i]

    eqns = s0[5,0:2] + x*ds[5,0:2] - s0[0,0:2] + y*ds[0,0:2]
    sol = sp.solve(eqns)
    s1[5] = s0[5] + sol[x]*ds[5]
    s1[6] = s1[0]

    # Plot MohrCoulomb yield criterion
    ax.plot(s1[:,0],s1[:,1],s1[:,2],'r-')

    # Plot points
    ax.plot( [sig[0]], [sig[1]], [sig[2]], marker='o', markersize=6, color="blue")
    tx = ('$\sigma_1$ = {:1.1f} MPa').format(sig[0]); ax.text(sig[0],sig[1],sig[2],tx)

    return ax

#=================================================================
#   PlotImplicit
#=================================================================

def plot_implicit(fn, bbox=(-2.5,2.5)):
    
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval'''

    ax = orthogonal_plot()
    
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3        
    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(xmin, xmax, 15) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y')

    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x')

    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)

    return ax


#=================================================================
#   Examples
#=================================================================


if __name__ == "__main__":


    fy = 5
    c = 0.3
    phi = np.pi*5/180
    
    ax = plot_DruckerPrager(None,c,phi,fy)
    plt.show()


    ax = plot_VonMises(None,fy)
    plt.show()


    ax = plot_implicit(calc_VonMises)
    plt.show()
