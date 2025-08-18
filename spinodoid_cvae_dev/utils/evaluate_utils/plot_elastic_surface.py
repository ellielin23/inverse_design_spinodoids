import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

blue = '#00305d'
red  = '#8b1a1a'
grey = '#717778'
index_map = {'11': 0,
             '22': 1,
             '33': 2,
             '12': 3,
             '13': 4,
             '23': 5,
             '1' : (0,0),
             '2' : (1,1),
             '3' : (2,2),
             '4' : (0,1),
             '5' : (0,2),
             '6' : (1,2)}

def mandel_to_tensor4(T_M):
    i = index_map
    T1111 = T_M[...,i['11'],i['11']]
    T1122 = T_M[...,i['11'],i['22']]
    T1133 = T_M[...,i['11'],i['33']]
    T1123 = T_M[...,i['11'],i['23']]/(2**0.5)
    T1113 = T_M[...,i['11'],i['13']]/(2**0.5)
    T1112 = T_M[...,i['11'],i['12']]/(2**0.5)
    T2222 = T_M[...,i['22'],i['22']]
    T2233 = T_M[...,i['22'],i['33']]
    T2223 = T_M[...,i['22'],i['23']]/(2**0.5)
    T2213 = T_M[...,i['22'],i['13']]/(2**0.5)
    T2212 = T_M[...,i['22'],i['12']]/(2**0.5)
    T3333 = T_M[...,i['33'],i['33']]
    T3323 = T_M[...,i['33'],i['23']]/(2**0.5)
    T3313 = T_M[...,i['33'],i['13']]/(2**0.5)
    T3312 = T_M[...,i['33'],i['12']]/(2**0.5)
    T2323 = T_M[...,i['23'],i['23']]/2
    T2313 = T_M[...,i['23'],i['13']]/2
    T2312 = T_M[...,i['23'],i['12']]/2
    T1313 = T_M[...,i['13'],i['13']]/2
    T1312 = T_M[...,i['13'],i['12']]/2
    T1212 = T_M[...,i['12'],i['12']]/2
    T1211 = T1112
    T1213 = T1312
    T1222 = T2212
    T1223 = T2312
    T1233 = T3312
    T1311 = T1113
    T1322 = T2213
    T1323 = T2313
    T1333 = T3313
    T2111 = T1112
    T2112 = T1212
    T2113 = T1312
    T2122 = T2212
    T2123 = T2312
    T2133 = T3312
    T2211 = T1122
    T2311 = T1123
    T2322 = T2223
    T2333 = T3323
    T3111 = T1113
    T3112 = T1312
    T3113 = T1313
    T3122 = T2213
    T3123 = T2313
    T3133 = T3313
    T3211 = T1123
    T3212 = T2312
    T3213 = T2313
    T3222 = T2223
    T3223 = T2323
    T3233 = T3323
    T3311 = T1133
    T3322 = T2233
    return tf.einsum('ijkl...->...ijkl', tf.convert_to_tensor(
                            [[[[T1111, T1112, T1113],
                               [T1112, T1122, T1123],
                               [T1113, T1123, T1133]],
                              [[T1211, T1212, T1213],
                               [T1212, T1222, T1223],
                               [T1213, T1223, T1233]],
                              [[T1311, T1312, T1313],
                               [T1312, T1322, T1323],
                               [T1313, T1323, T1333]]],

                             [[[T2111, T2112, T2113],
                               [T2112, T2122, T2123],
                               [T2113, T2123, T2133]],
                              [[T2211, T2212, T2213],
                               [T2212, T2222, T2223],
                               [T2213, T2223, T2233]],
                              [[T2311, T2312, T2313],
                               [T2312, T2322, T2323],
                               [T2313, T2323, T2333]]],

                             [[[T3111, T3112, T3113],
                               [T3112, T3122, T3123],
                               [T3113, T3123, T3133]],
                              [[T3211, T3212, T3213],
                               [T3212, T3222, T3223],
                               [T3213, T3223, T3233]],
                              [[T3311, T3312, T3313],
                               [T3312, T3322, T3323],
                               [T3313, T3323, T3333]]]],
                            ))

def tensor4_to_mandel(T):
    i   = index_map
    k11 = T[...,*i['1'],*i['1']]
    k12 = T[...,*i['1'],*i['2']]
    k13 = T[...,*i['1'],*i['3']]
    k14 = T[...,*i['1'],*i['4']]*2**(0.5)
    k15 = T[...,*i['1'],*i['5']]*2**(0.5)
    k16 = T[...,*i['1'],*i['6']]*2**(0.5)
    k22 = T[...,*i['2'],*i['2']]
    k23 = T[...,*i['2'],*i['3']]
    k24 = T[...,*i['2'],*i['4']]*2**(0.5)
    k25 = T[...,*i['2'],*i['5']]*2**(0.5)
    k26 = T[...,*i['2'],*i['6']]*2**(0.5)
    k33 = T[...,*i['3'],*i['3']]
    k34 = T[...,*i['3'],*i['4']]*2**(0.5)
    k35 = T[...,*i['3'],*i['5']]*2**(0.5)
    k36 = T[...,*i['3'],*i['6']]*2**(0.5)
    k44 = T[...,*i['4'],*i['4']]*2
    k45 = T[...,*i['4'],*i['5']]*2
    k46 = T[...,*i['4'],*i['6']]*2
    k55 = T[...,*i['5'],*i['5']]*2
    k56 = T[...,*i['5'],*i['6']]*2
    k66 = T[...,*i['6'],*i['6']]*2

    return tf.einsum('ij...->...ij', tf.convert_to_tensor(
        [[k11, k12, k13, k14, k15, k16],
         [k12, k22, k23, k24, k25, k26],
         [k13, k23, k33, k34, k35, k36],
         [k14, k24, k34, k44, k45, k46],
         [k15, k25, k35, k45, k55, k56],
         [k16, k26, k36, k46, k56, k66]]))


def plot_elastic_surface(C, save_as=None):
    
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        # x_limits = (-1,1)
        # y_limits = (-1,1)
        # z_limits = (-1,1)

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def sample_direction_vectors():
        ds = 0.01
        r_phi = np.arange(0.,2*np.pi+ds,ds)
        r_theta = np.arange(0.,np.pi+ds,ds)
        N1_plt = np.einsum("i,j->ij",np.cos(r_phi), np.sin(r_theta))
        N2_plt = np.einsum("i,j->ij",np.sin(r_phi), np.sin(r_theta))
        N3_plt = np.einsum("i,j->ij",np.ones([r_phi.shape[0]],dtype=r_phi.dtype), np.cos(r_theta))
        N1 = np.reshape(N1_plt, [np.prod(N1_plt.shape)])
        N2 = np.reshape(N2_plt, [np.prod(N2_plt.shape)])
        N3 = np.reshape(N3_plt, [np.prod(N3_plt.shape)])
        N = np.stack([N1,N2,N3], axis=-1)
        return N, [r_phi.shape[0], r_theta.shape[0]]

    d, shapes = sample_direction_vectors()
    S = mandel_to_tensor4(np.linalg.inv(tensor4_to_mandel(C)))
    E = 1 / (np.einsum("ijkl,ni,nj,nk,nl->n", S, d,d,d,d))
    surface_points = d*np.reshape(E, [-1,1])

    EN1 = np.reshape(surface_points[:,0], [shapes[0], shapes[1]])
    EN2 = np.reshape(surface_points[:,1], [shapes[0], shapes[1]])
    EN3 = np.reshape(surface_points[:,2], [shapes[0], shapes[1]])
    E = E.reshape(EN1.shape)

    # norm = mpl.colors.Normalize(vmin=np.min(E), vmax=np.max(E))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=0.5)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    # cm = plt.cm.get_cmap(cmap)
    cm = mcolors.LinearSegmentedColormap.from_list('CustomBlueRed', [
        (0,blue),
        (0.3, grey),
        (1,red),
        ])
    ax.plot_surface(EN1, EN2, EN3, 
                    rcount=360, ccount=360,
                    facecolors=cm(norm(E)), shade=False, )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    set_axes_equal(ax)
    ax.set_axis_off()
    m = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    m.set_array([])    
    # colorbar
    fig.subplots_adjust(right=0.8, hspace=0.25)
    cax = fig.add_axes((0.8, 0.2, 0.03, 0.6))
    l = r"$E$" 
    fig.colorbar(m, cax=cax, label=l)

    ax.view_init(elev=20, azim=-110)
    if save_as:
        plt.savefig(save_as, dpi=600)
    plt.show()