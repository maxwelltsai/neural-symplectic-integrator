import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import os 
import argparse
import h5py


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

def fading_line(x, y, color='black', alpha=1, fading=True, fancy=False, tail=100, linestyle='solid', **kwargs):
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

    lc = LineCollection(segments, color=colors, linestyle=linestyle, **kwargs)
    return lc


def multicolored_fading_lines(x, y, tail=100, ax=None, marker='o', s=plt.rcParams['lines.markersize']**2, alpha=1, label=None, linestyle=None):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """

    norm = plt.Normalize(vmin=0, vmax=x.shape[1])
    
    colors = []
    for i in range(x.shape[1]):
        rgba_color = plt.cm.jet(norm(i), bytes=False) 
        colors.append(rgba_color)
        lc = fading_line(x[:,i], y[:,i], color=rgba_color, tail=tail, linestyle=linestyle)
        ax.add_collection(lc)
    ax.scatter(x[-1], y[-1], cmap=plt.cm.jet, norm=norm, c=colors, marker=marker, s=s, alpha=alpha, label=label)
    ax.set_xlim(x.min()*1.1, x.max()*1.1)
    ax.set_ylim(y.min()*1.1, y.max()*1.1)
    return ax 


if __name__ == '__main__':
    from config import CONFIG 

    parser = argparse.ArgumentParser()
    parser.add_argument('--wh', type=str, dest='wh', default='data_nb.h5', help='Name of the output data file created by a traditional WH integrator')
    parser.add_argument('--nih', type=str, dest='nih', default='data_nih_MLP_SymmetricLog.h5', help='Name of the output data file created by a traditional WH integrator')
    parser.add_argument('-o', type=str, dest='mp4_fn', default='out.mp4', help='File name of the output movie')
    parser.add_argument('-r', '--framerate', type=int, dest='framerate', default=20, help='Framerate per second')
    parser.add_argument('-x', type=str, dest='x', default='x', help='The name of the x-axis to plot')
    parser.add_argument('-y', type=str, dest='y', default='y', help='The name of the y-axis to plot')
    parser.add_argument('-t', '--tail', type=int, dest='tail', default=200, help='Length of the tail')
    args = parser.parse_args()

    with h5py.File(args.nih, 'r') as h5f:
        step_id = 0
        ecc_hat = h5f['Step#%d/ecc' % step_id][()]
        semi_hat = h5f['Step#%d/a' % step_id][()]
        x_hat = h5f['Step#%d/%s' % (step_id, args.x)][()]
        y_hat = h5f['Step#%d/%s' % (step_id, args.y)][()]

    with h5py.File(args.wh, 'r') as h5f:
        step_id = 0
        ecc = h5f['Step#%d/ecc' % step_id][()]
        semi = h5f['Step#%d/a' % step_id][()]
        x = h5f['Step#%d/%s' % (step_id, args.x)][()]
        y = h5f['Step#%d/%s' % (step_id, args.y)][()]
        t = h5f['Step#%d/time' % step_id][()]

    if not os.path.isdir(CONFIG['fig_dir']):
        os.mkdir(CONFIG['fig_dir'])

    for i in range(1, min(x.shape[0], x_hat.shape[0])):
        print('Step#%d' % i)
        fig, ax = plt.subplots(figsize=(10,10))
        plt.axis('equal')
        lim_low = max(0, i-args.tail)
        ax = multicolored_fading_lines(x[lim_low:i], y[lim_low:i], ax=ax, tail=args.tail, marker='s', s=2*plt.rcParams['lines.markersize']**2, alpha=0.5, linestyle='dashed', label='WH')
        ax = multicolored_fading_lines(x_hat[lim_low:i], y_hat[lim_low:i], ax=ax, tail=args.tail, label='WH-NIH', linestyle='solid')

        # plot the difference
        for j in range(x.shape[1]):
            plt.annotate(text='', xy=(x[i,j], y[i,j]), xytext=(x_hat[i,j], y_hat[i,j]), arrowprops=dict(arrowstyle='<->'), alpha=0.7)

        plt.title('Step#%d, $t = $%f yrs' % (i, t[i]))
        plt.legend()
        plt.savefig(os.path.join(CONFIG['fig_dir'], 'step%05d.png' % i))
        plt.close()

    # Combine the frames into a movie
    cmd = 'ffmpeg -y -framerate {} -start_number 1 -i {}/step%05d.png -c:v libx264 -r 20 -pix_fmt yuv420p {}'.format(args.framerate, CONFIG['fig_dir'], args.mp4_fn)
    print('Creating movie with the following command: %s' % cmd)
    os.system(cmd)