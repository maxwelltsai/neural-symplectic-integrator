import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib
import os 

def get_color(color):
    """
    Takes a string for a color name defined in matplotlib and returns of a 3-tuple of RGB values.
    Will simply return passed value if it's a tuple of length three.
    Parameters
    ----------
    color   : str
        Name of matplotlib color to calculate RGB values for.
    """

    if isinstance(color, tuple) and len(color) == 3: # already a tuple of RGB values
        return color
    elif isinstance(color, tuple) and len(color) == 4: # already a tuple of RGB values
        return (color[0], color[1], color[2])

    try:
        import matplotlib.colors as mplcolors
    except:
        raise ImportError("Error importing matplotlib. If running from within a jupyter notebook, try calling '%matplotlib inline' beforehand.")
   
    try:
        hexcolor = mplcolors.cnames[color]
    except KeyError:
        raise AttributeError("Color not recognized in matplotlib.")

    hexcolor = hexcolor.lstrip('#')
    lv = len(hexcolor)
    return tuple(int(hexcolor[i:i + lv // 3], 16)/255. for i in range(0, lv, lv // 3)) # tuple of rgb values

def fading_line(x, y, color='black', alpha=1, fading=True, fancy=False, tail=100, **kwargs):
    """
    Returns a matplotlib LineCollection connecting the points in the x and y lists.
    Can pass any kwargs you can pass to LineCollection, like linewidgth.
    Parameters
    ----------
    x       : list or array of floats for the positions on the (plot's) x axis.
    y       : list or array of floats for the positions on the (plot's) y axis.
    color   : Color for the line. 3-tuple of RGB values, hex, or string. Default: 'black'.
    alpha   : float, alpha value of the line. Default 1.
    fading  : bool, determines if the line is fading along the orbit.
    fancy   : bool, same as fancy argument in OrbitPlot()
    """
    try:
        from matplotlib.collections import LineCollection
        import numpy as np
    except:
        raise ImportError("Error importing matplotlib and/or numpy. Plotting functions not available. If running from within a jupyter notebook, try calling '%matplotlib inline' beforehand.")


    if "lw" not in kwargs:
        kwargs["lw"] = 1
    lw = kwargs["lw"]

    if fancy:
        kwargs["lw"] = 1*lw
        fl1 = fading_line(x, y, color=color, alpha=alpha, fading=fading, fancy=False, **kwargs)
        kwargs["lw"] = 2*lw
        alpha *= 0.5
        fl2 = fading_line(x, y, color=color, alpha=alpha, fading=fading, fancy=False, **kwargs)
        kwargs["lw"] = 6*lw
        alpha *= 0.5
        fl3 = fading_line(x, y, color=color, alpha=alpha, fading=fading, fancy=False, **kwargs)
        return [fl3,fl2,fl1]
    
    Npts = min(tail, len(x))
    x = x[len(x)-Npts:]
    y = y[len(y)-Npts:]
    Npts = len(x)
    if len(y) != Npts:
        raise AttributeError("x and y must have same dimension.")
    
    color = get_color(color)
    colors = np.zeros((Npts,4))
    colors[:,0:3] = color
    if fading:
        colors[:,3] = alpha*np.linspace(0,1,Npts)
    else:
        colors[:,3] = alpha
   
    segments = np.zeros((Npts-1,2,2))

    segments[:,0,0] = x[:-1]
    segments[:,0,1] = y[:-1]
    segments[:,1,0] = x[1:]
    segments[:,1,1] = y[1:]

    lc = LineCollection(segments, color=colors, **kwargs)
    return lc


def multicolored_fading_lines(x, y, tail=100, ax=None):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """

    norm = plt.Normalize(vmin=0, vmax=x.shape[1])
    
    colors = []
    for i in range(x.shape[1]):
        rgba_color = plt.cm.jet(norm(i), bytes=False) 
        colors.append(rgba_color)
        lc = fading_line(x[:,i], y[:,i], color=rgba_color, tail=tail)
        ax.add_collection(lc)
    ax.scatter(x[-1], y[-1], cmap=plt.cm.jet, norm=norm, c=colors)
    ax.set_xlim(x.min()*1.1, x.max()*1.1)
    ax.set_ylim(y.min()*1.1, y.max()*1.1)
    return ax 


if __name__ == '__main__':
    import h5py 
    with h5py.File('data_nih.h5', 'r') as h5f:
        step_id = 1
        ecc_hat = h5f['Step#%d/ecc' % step_id][()]
        semi_hat = h5f['Step#%d/a' % step_id][()]
        x_hat = h5f['Step#%d/x' % step_id][()]
        y_hat = h5f['Step#%d/y' % step_id][()]
        z_hat = h5f['Step#%d/z' % step_id][()]

    with h5py.File('data_nb.h5', 'r') as h5f:
        step_id = 1
        ecc = h5f['Step#%d/ecc' % step_id][()]
        semi = h5f['Step#%d/a' % step_id][()]
        x = h5f['Step#%d/x' % step_id][()]
        y = h5f['Step#%d/y' % step_id][()]
        z = h5f['Step#%d/z' % step_id][()]
    
    for i in range(5, x_hat.shape[0]):
        print('Step#%d' % i)
        fig, ax = plt.subplots(figsize=(10,10))
        plt.axis('equal')
        ax = multicolored_fading_lines(x[:i], y[:i], ax=ax, tail=200)
        plt.savefig(os.path.join('figures', 'step%05d.png' % i))
        plt.close()