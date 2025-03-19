from mpl_toolkits.mplot3d import Axes3D 
from umap import UMAP
import colorsys
from colour import Color
import numpy as np
import matplotlib.pyplot as plt

def umap_reduction(data, n_components=3, min_dist=0.1, n_neighbors=15, metric='euclidean'):
    print('UMAP reduction...')
    embedding = np.empty((data.shape[0], data.shape[1], n_components))
    for bId in range(data.shape[0]):
        reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
        embedding[bId] = reducer.fit_transform(data[bId])
    return embedding,reducer

def plot_3d_polar(polar_coord):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter(polar_coord[:,0], polar_coord[:,1], polar_coord[:,2])
    plt.show()

def find_zero_point(x, y, z):
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 0 and z[i] == 0:
            return i
    return None

def nd_to_3d_polar(data):
    # Step 2: Convert each point to polar coordinates
    polar_coords = np.empty(data.shape)
    for bId in range(data.shape[0]):
        for n_point in range(data.shape[1]):
            x, y, z = data[bId, n_point]
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arccos(z / r)
            phi = np.arctan2(y, x)
            polar_coords[bId,n_point] = [r, theta, phi]
    
    return polar_coords

def traslate_eye_to_zero(cartesian_coord):
    translated = np.empty(cartesian_coord.shape)
    for bId in range(cartesian_coord.shape[0]):
        eye = cartesian_coord[bId, 0]
        translated[bId] = cartesian_coord[bId] - eye
    return translated

def whitening_around_eye(pca_points):
    whitened = np.empty(pca_points.shape)
    for bId in range(pca_points.shape[0]):
        translated = traslate_eye_to_zero(pca_points)

        # Step 2: Rotate the system to align the chosen points to the xy-plane
        # Calculate the normal of the plane defined by the chosen points
        ref1 = pca_points[bId,-2] - np.eye[0]
        ref2 = pca_points[bId,-1] - np.eye[0]
        normal = np.cross(ref1, ref2)
        normal = normal / np.linalg.norm(normal)

        # Calculate the rotation matrix to align the normal vector to the z-axis
        z_axis = np.array([0, 0, 1])
        v = np.cross(normal, z_axis)
        s = np.linalg.norm(v)
        c = np.dot(normal, z_axis)
        I = np.eye(3)
        if s != 0:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = I + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
        else:
            R = I  # No rotation needed if normal is already aligned with z_axis

        aligned_points = np.dot(R, translated.T).T

        # Step 3: Rotate around the z-axis to place the second chosen point on the x-axis
        # Find the angle to rotate the second chosen point to the x-axis
        angle = np.arctan2(aligned_points[-2, 1], aligned_points[-2, 0])
        rotation_z = np.array([
            [np.cos(-angle), -np.sin(-angle), 0],
            [np.sin(-angle), np.cos(-angle), 0],
            [0, 0, 1]
        ])
        whitened[bId] = np.dot(rotation_z, aligned_points.T).T
    return whitened

def generate_equally_distributed_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        lightness = 0.4
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        rgb = tuple(x for x in rgb)
        colors.append(rgb)
    return colors

def get_color_gradient(start_color, end_color, n):
    start_color = Color(start_color)
    end_color = Color(end_color)
    gradient = list(start_color.range_to(end_color, n))
    return [c.get_rgb() for c in gradient]

def plot_onVector(ax, cartesian_coord, vector, size=None, legend=True, markerscale=2):
    color = np.zeros((cartesian_coord.shape[0],3))
    sessions = np.unique(vector[~np.isnan(vector)])
    nsession = len(sessions)
    random_colors = generate_equally_distributed_colors(nsession)
    idx_nan = np.where(np.isnan(vector))[0]

    if size is None:
        size = np.array([5]*cartesian_coord.shape[0])

    if cartesian_coord.shape[1] == 2:
        if len(idx_nan) > 0:
            ax.scatter(cartesian_coord[idx_nan,0], cartesian_coord[idx_nan,1], c=color[idx_nan], s=size[idx_nan])
        for idx_ses in range(nsession):
            idx = np.where(vector == sessions[idx_ses])[0]
            clr = np.tile(random_colors[idx_ses], (len(idx), 1))
            ax.scatter(cartesian_coord[idx,0], cartesian_coord[idx,1], c=clr, s=size[idx], label=sessions[idx_ses])

    elif cartesian_coord.shape[1] == 3:
        if len(idx_nan) > 0:
            ax.scatter(cartesian_coord[idx_nan,0], cartesian_coord[idx_nan,1], cartesian_coord[idx_nan,2], c=color[idx_nan], s=size[idx_nan])
        for idx_ses in range(nsession):
            idx = np.where(vector == sessions[idx_ses])[0]
            clr = np.tile(random_colors[idx_ses], (len(idx), 1))
            ax.scatter(cartesian_coord[idx,0], cartesian_coord[idx,1], cartesian_coord[idx,2], c=clr, s=size[idx], label=sessions[idx_ses])
            ax.set_zlabel('Z')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if legend:
        ax.legend(markerscale=markerscale)

def plot_gradientOnVector(ax, cartesian_coord, vector, size=None, colormap=None):
    trial_max = int(max(vector[~np.isnan(vector)]))

    if size is None:
        size = 30*np.ones(cartesian_coord.shape[0])

    if colormap is None:
        color = np.zeros((cartesian_coord.shape[0],3))
        color_gradient = get_color_gradient('red', 'green', trial_max)
        for i,val in enumerate(vector):
            if not np.isnan(val):
                color[i] = color_gradient[int(val)-1]
    else:
        color = np.zeros((cartesian_coord.shape[0],4))
        colormap = plt.get_cmap(colormap)
        for i in range(len(color) - 1):
            color[i] = colormap(i / (len(color) - 2))
        

    if cartesian_coord.shape[1] == 2:
        ax.scatter(cartesian_coord[:,0], cartesian_coord[:,1], c=color, s=size)

    elif cartesian_coord.shape[1] == 3:
        ax.scatter(cartesian_coord[:,0], cartesian_coord[:,1], cartesian_coord[:,2], c=color, s=size)
        ax.set_zlabel('Z')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

def plot_onClasses(ax, cartesian_coord, labels, size=None):

    color = np.zeros((cartesian_coord.shape[0],3))
    if labels is not None:
        classes = np.unique(labels)
        random_colors = np.array(generate_equally_distributed_colors(len(classes)))
        for i,cl in enumerate(classes):
            color[np.squeeze(labels == cl)] = random_colors[i]
        
    if size is None:
        size = 30*np.ones(cartesian_coord.shape[0])

    if cartesian_coord.shape[1] == 2:
        ax.scatter(cartesian_coord[:,0], cartesian_coord[:,1], c=color, s=size)

    elif cartesian_coord.shape[1] == 3:
        ax.scatter(cartesian_coord[:,0], cartesian_coord[:,1], cartesian_coord[:,2], c=color, s=size)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if cartesian_coord.shape[1] == 3:
        ax.set_zlabel('Z')
        
def plot_cartesian(cartesian_coord, labels=None, sessionVector=None, trialsGradientVector=None, fig_size=(16,5), point_size=5):

    num_subplots = 1
    if sessionVector is not None:
        num_subplots += 1
    if trialsGradientVector is not None:
        num_subplots += 1
    
    size = np.array([point_size]*cartesian_coord.shape[0])
    if labels is not None:
        size[np.isnan(labels)] = [30]*sum(np.isnan(labels))

    
    fig = plt.figure(figsize=fig_size)
    ax = fig.gca()


    actual_subplot = 1
    if cartesian_coord.shape[1] == 2:
        ax = fig.add_subplot(1,num_subplots,actual_subplot)
    elif cartesian_coord.shape[1] == 3: 
        ax = fig.add_subplot(1,num_subplots,actual_subplot, projection ='3d')
    plot_onClasses(ax, cartesian_coord, labels, size=size)


    if sessionVector is not None:
        actual_subplot += 1
        if cartesian_coord.shape[1] == 2:
            ax = fig.add_subplot(1,num_subplots,actual_subplot)
        elif cartesian_coord.shape[1] == 3: 
            ax = fig.add_subplot(1,num_subplots,actual_subplot, projection ='3d')
        plot_onVector(ax, cartesian_coord, sessionVector, size=size)



    if trialsGradientVector is not None:
        actual_subplot += 1
        if cartesian_coord.shape[1] == 2:
            ax = fig.add_subplot(1,num_subplots,actual_subplot)
        elif cartesian_coord.shape[1] == 3: 
            ax = fig.add_subplot(1,num_subplots,actual_subplot, projection ='3d')
        plot_gradientOnVector(ax, cartesian_coord, trialsGradientVector, size=size)

    plt.show()


###_______________________________________________________________________________________________________________________###

 
def polarPlot_centroids(distance, angles, point_size=None, max_distance=None, marker_type=None, day_vector=None, do_angle_scaling=False, bandranges=None, idx_stopRec=None, idx_recal=None, figsize=(17, 10)):
    
    n_bands, n_centroids, n_classes = distance.shape

    fig, axs = plt.subplots(n_classes, n_bands, subplot_kw={'projection': 'polar'}, figsize=figsize)

    if point_size is None:
        point_size = np.ones((n_bands, n_centroids, n_classes)) * 25
    elif len(point_size.shape)==1:
        # np.tile(np.eye(cl_data.shape[-1]), (np.prod(cl_data.shape[0:-2]), 1, 1)).reshape(cl_data.shape)
        t_point = point_size
        point_size = np.zeros((n_bands, n_centroids, n_classes))
        for i in range(n_bands):
            for j in range(n_classes):
                point_size[i,:,j] = t_point

    if max_distance is None:
        max_distance = np.max(distance)

    if day_vector is None:
        day_vector = np.ones(n_centroids)

    if marker_type is None:
        marker_type = ['o']*len(np.unique(day_vector))


    for cl in range(n_classes):
        for b in range(n_bands):
            angl = np.deg2rad(angles[b,:,cl])
            colors = plt.cm.RdYlGn(np.linspace(0, 1, len(angl)))

            if do_angle_scaling:
                angl = 2*angl

            for k in np.unique(day_vector):
                idx = day_vector == k
                axs[cl, b].scatter(angl[idx], distance[b,idx,cl], s=point_size[b,idx,cl] , color=colors[idx], marker=marker_type[int(k-1)], edgecolors='k')

            if idx_stopRec is not None:
                axs[cl, b].scatter(angl[idx_stopRec], distance[b,idx_stopRec,cl], s=point_size[b,idx_stopRec,cl], marker='X', color='black', edgecolor=colors[idx_stopRec])
            if idx_recal is not None:
                axs[cl, b].scatter(angl[idx_recal], distance[b,idx_recal,cl], s=point_size[b,idx_stopRec,cl], marker='D', color='black', edgecolor=colors[idx_recal])
 
            axs[cl, b].scatter(angl[0], distance[b,0,cl], s=point_size[b,0,cl], c='k')

            if b == 0:
                axs[cl, b].set_ylabel(f'Class : {cl}', fontweight='bold', labelpad=30, fontsize=15)

            if cl == 0:
                axs[cl, b].set_title(['Band no. '+str(b) if bandranges is None else 'Band '+str(bandranges[b])][0], fontweight='bold', fontsize=14, pad=10)

            axs[cl, b].set_rlabel_position(-22.5)  # Move radial labels away from plotted line
            if do_angle_scaling:
                axs[cl, b].set_thetalim(0, np.pi)
                tick_locations = np.linspace(0, np.pi, 7)  # Define new tick locations
                tick_labels = [f"{np.rad2deg(t/2):.0f}" for t in tick_locations]  # Scale labels back to [0, pi/2]
                axs[cl, b].set_xticks(tick_locations)
                axs[cl, b].set_xticklabels(tick_labels)
            else:
                axs[cl, b].set_thetalim(0, np.pi/2)  # Set theta limits to plot only the first and second quarters
            axs[cl, b].set_rlim(0, max_distance)  # Set radius limits

    plt.tight_layout()
    plt.show()
    


def plot_centroids_movement(data, classes, dates=None, x_dates=None, max_value=None, accuracy=None, x_accuracy=None, title='', bandranges=None, step_dates=1, stop_idx=[], rec_idx=[], pointStd=None, figsize=(17,4)):
    # data = bands x runs x classes 
    # classes = list of classes
    if max_value is None:
        max_value = np.max(data)
    if x_dates is None:
        x_dates = range(0,data.shape[1],step_dates)
    if dates is None:
        dates = range(data.shape[1])

    plt.figure(figsize=figsize)
    for bId in range(data.shape[0]):
        plt.subplot(1,2,bId+1)
        ax = plt.gca()
        if accuracy is not None:
            ax2 = ax.twinx() 
            ax.bar(x_accuracy, accuracy, label='Accuracy', color='blue', alpha=0.08)
            ax.set_ylim(0,1)
        else:
            ax2 = ax
        for rec in stop_idx:
            ax2.axvline(x = rec, color = 'r')
        for rec in rec_idx:
            ax2.axvline(x = rec, color = 'g')

        if pointStd is None:
            ax2.plot(data[bId], label=classes)
        else:
            for cl in range(data.shape[2]):
                ax2.errorbar(range(data.shape[1]), data[bId,:,cl], yerr=pointStd[bId,:,cl], label=classes[cl])

        ax2.legend()
        ax2.set_ylim(0,max_value*1.05) 
        ax2.set_xlim(-1,data.shape[1])
        ax.set_xticks(x_dates)
        ax.set_xticklabels(dates[::step_dates], rotation=90, fontsize=8)
        plt.title(['Band no. '+str(bId) if bandranges is None else 'Band '+str(bandranges[bId])][0])
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()


def plot_centroids_angles(data, classes, suptitle=None, titles=None, cmap='YlGn', interpolation=None, normalize=False, dates=None, x_dates=None, max_value=None, min_value=None, title='', bandranges=None, step_dates=1, stop_idx=[], rec_idx=[], figsize=(17,4)):
    # data = bands x runs x classes 
    # classes = list of classes
  
# Create subplots (2x2)
    
    if normalize == True:
        min_angles = np.min(data, axis=-1,keepdims=True)
        max_angles = np.max(data, axis=-1,keepdims=True)
        data = (data - min_angles) / (max_angles - min_angles)

    if max_value is None:
        max_value = np.max(data)
    if min_value is None:
        min_value = np.min(data)

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    for idxB in range(data.shape[0]):
        for idxCl in range(len(classes)):
            for rec in stop_idx:
                axs[idxCl, idxB].axvline(x = rec, color = 'r')
            for rec in rec_idx:
                axs[idxCl, idxB].axvline(x = rec, color = 'g')
            im = axs[idxCl, idxB].imshow(data[idxB,:,idxCl].T, aspect='auto', cmap=cmap, vmin=min_value, vmax=max_value, interpolation=interpolation)
            if dates is not None and x_dates is not None:
                axs[idxCl, idxB].set_xticks(x_dates)
                axs[idxCl, idxB].set_xticklabels(dates[::step_dates], rotation=90, fontsize=8)
            else:
                axs[idxCl, idxB].set_xticks([])
                axs[idxCl, idxB].set_xticklabels([])
                axs[idxCl, idxB].set_yticks([])
                axs[idxCl, idxB].set_yticklabels([])
            if titles is None:
                if idxCl == 0:
                    axs[idxCl, idxB].set_title(['Band no. '+str(idxB) if bandranges is None else 'Band '+str(bandranges[idxB])][0])
            else:
                axs[idxCl, idxB].set_title(titles[idxCl,idxB])
    
    for ax in axs.flat:
        ax.label_outer()

    fig.subplots_adjust(hspace=0.2, wspace=0.12, top=0.85, left=0.03, right=0.995)

    cbar_ax = fig.add_axes([0.25, 0.95, 0.5, 0.02])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    if suptitle is not None:
        cbar_ax.set_title(suptitle)

    # plt.tight_layout(rect=[0, 0, 1, 0.9]) 
    plt.show()
