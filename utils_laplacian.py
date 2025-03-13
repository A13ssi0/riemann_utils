from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt


class EEG_CHANNEL:
    def __init__(self, label, theta, radius, x, y, z, sph_theta, sph_phi, sph_radius):
        self.label = label
        self.theta = theta
        self.radius = radius
        self.x = x
        self.y = y
        self.z = z
        self.sph_theta = sph_theta
        self.sph_phi = sph_phi
        self.sph_radius = sph_radius
        

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def get_standard_chanlocs():
    chanlocs = np.array([
    EEG_CHANNEL('FP1',-19.3, 0.525, 83.9, 29.4, -6.99, 19.3, -4.49, 89.2),
    EEG_CHANNEL('FPZ',0.0729, 0.506, 88.2, -0.112, -1.71, -0.0729, -1.11, 88.3),
    EEG_CHANNEL('FP2',19.4, 0.525, 84.9, -29.9, -7.08, -19.4, -4.50, 90.3),
    EEG_CHANNEL('F7',-58.8, 0.544, 42.5, 70.3, -11.4, 58.8, -7.92, 82.9),
    EEG_CHANNEL('F3',-43.4, 0.333, 53.1, 50.2, 42.2, 43.4, 30, 84.4),
    EEG_CHANNEL('FZ',0.306, 0.230, 58.5, -0.312, 66.5, -0.306, 48.6, 88.5),
    EEG_CHANNEL('F4',43.7, 0.341, 54.3, -51.8, 40.8, -43.7, 28.5, 85.5),
    EEG_CHANNEL('F8',58.7, 0.544, 44.4, -73, -12, -58.7, -7.99, 86.3),
    EEG_CHANNEL('FC5',-76.4, 0.405, 18.6, 77.2, 24.5, 76.4, 17.1, 83.1),
    EEG_CHANNEL('FC1',-52.6, 0.157, 26, 34.1, 80, 52.6, 61.8, 90.7),
    EEG_CHANNEL('FC2',52.8, 0.161, 26.4, -34.8, 78.8, -52.8, 61, 90.1),
    EEG_CHANNEL('FC6',75.9, 0.408, 19.9, -79.5, 24.4, -75.9, 16.6, 85.6),
    EEG_CHANNEL('M1',-118, 0.694, -45, 86.1, -68, 118, -35, 119),
    EEG_CHANNEL('T7',-101, 0.535, -16, 84.2, -9.35, 101, -6.23, 86.2),
    EEG_CHANNEL('C3',-100, 0.255, -11.6, 65.4, 64.4, 100, 44.1, 92.5),
    EEG_CHANNEL('CZ',177, 0.0291, -9.17, -0.401, 100, -177, 84.8, 101),
    EEG_CHANNEL('C4',99.2, 0.261, -10.9, -67.1, 63.6, -99.2, 43.1, 93.1),
    EEG_CHANNEL('T8',100, 0.535, -15, -85.1, -9.49, -100, -6.27, 86.9),
    EEG_CHANNEL('M2',118, 0.695, -45, -85.8, -68, -118, -35.1, 118),
    EEG_CHANNEL('CP5',-120, 0.397, -46.6, 79.6, 30.9, 120, 18.6, 97.3),
    EEG_CHANNEL('CP1',-143, 0.183, -47.3, 35.5, 91.3, 143, 57.1, 109),
    EEG_CHANNEL('CP2',141, 0.188, -47.1, -38.4, 90.7, -141, 56.2, 109),
    EEG_CHANNEL('CP6',119, 0.399, -46.1, -83.3, 31.2, -119, 18.1, 100),
    EEG_CHANNEL('P7',-135, 0.508, -73.5, 72.4, -2.49, 135, -1.38, 103),
    EEG_CHANNEL('P3',-146, 0.331, -78.8, 53, 55.9, 146, 30.5, 110),
    EEG_CHANNEL('PZ',180, 0.247, -81.1, -0.325, 82.6, -180, 45.5, 116),
    EEG_CHANNEL('P4',145, 0.331, -78.6, -55.7, 56.6, -145, 30.4, 112),
    EEG_CHANNEL('P8',135, 0.508, -73.1, -73.1, -2.54, -135, -1.41, 103),
    EEG_CHANNEL('POZ',180, 0.354, -102, -0.216, 50.6, -180, 26.3, 114),
    EEG_CHANNEL('O1',-165, 0.476, -112, 29.4, 8.84, 165, 4.35, 117),
    EEG_CHANNEL('O2',165, 0.476, -112, -29.8, 8.80, -165, 4.34, 116),
    EEG_CHANNEL('AF7',-38.7, 0.538, 68.6, 54.8, -10.6, 38.7, -6.88, 88.4),
    EEG_CHANNEL('AF3',-23.7, 0.421, 76.8, 33.7, 21.2, 23.7, 14.2, 86.5),
    EEG_CHANNEL('AF4',24.7, 0.420, 77.7, -35.7, 22, -24.7, 14.4, 88.3),
    EEG_CHANNEL('AF8',38.7, 0.538, 69.7, -55.7, -10.8, -38.7, -6.87, 89.9),
    EEG_CHANNEL('F5',-53.3, 0.434, 48, 64.5, 16.9, 53.3, 11.9, 82.2),
    EEG_CHANNEL('F1',-25.8, 0.257, 56.9, 27.5, 60.3, 25.8, 43.7, 87.4),
    EEG_CHANNEL('F2',27.1, 0.263, 57.6, -29.5, 59.5, -27.1, 42.6, 87.9),
    EEG_CHANNEL('F6',53.7, 0.439, 49.8, -67.9, 16.4, -53.7, 11, 85.8),
    EEG_CHANNEL('FC3',-69.3, 0.273, 22.7, 60.2, 55.5, 69.3, 40.8, 85),
    EEG_CHANNEL('FCZ',0.787, 0.0954, 27.4, -0.376, 88.7, -0.787, 72.8, 92.8),
    EEG_CHANNEL('FC4',69.2, 0.279, 23.7, -62.3, 55.6, -69.2, 39.8, 86.8),
    EEG_CHANNEL('C5',-99.7, 0.391, -13.8, 80.3, 29.2, 99.7, 19.7, 86.5),
    EEG_CHANNEL('C1',-105, 0.126, -9.98, 36.2, 89.8, 105, 67.3, 97.3),
    EEG_CHANNEL('C2',104, 0.132, -9.62, -37.7, 88.4, -104, 66.3, 96.6),
    EEG_CHANNEL('C6',98.7, 0.394, -12.8, -83.5, 29.2, -98.7, 19.1, 89.3),
    EEG_CHANNEL('CP3',-126, 0.279, -47, 63.6, 65.6, 126, 39.7, 103),
    EEG_CHANNEL('CP4',125, 0.284, -46.6, -66.6, 65.6, -125, 38.9, 104),
    EEG_CHANNEL('P5',-139, 0.413, -76.3, 67.3, 28.4, 139, 15.6, 106),
    EEG_CHANNEL('P1',-160, 0.270, -80.5, 28.6, 75.4, 160, 41.4, 114),
    EEG_CHANNEL('P2',158, 0.269, -80.5, -31.9, 76.7, -158, 41.5, 116),
    EEG_CHANNEL('P6',138, 0.414, -75.9, -67.9, 28.1, -138, 15.4, 106),
    EEG_CHANNEL('PO5',-154, 0.439, -99.3, 48.4, 21.6, 154, 11.1, 113),
    EEG_CHANNEL('PO3',-160, 0.394, -101, 36.5, 37.2, 160, 19.1, 114),
    EEG_CHANNEL('PO4',160, 0.396, -101, -36.8, 36.4, -160, 18.7, 113),
    EEG_CHANNEL('PO6',153, 0.439, -99.4, -49.8, 21.7, -153, 11.1, 113),
    EEG_CHANNEL('FT7',-80.1, 0.543, 14.1, 80.8, -11.1, 80.1, -7.73, 82.8),
    EEG_CHANNEL('FT8',79.3, 0.543, 15.4, -81.8, -11.3, -79.3, -7.75, 84),
    EEG_CHANNEL('TP7',-118, 0.523, -46, 84.8, -7.06, 118, -4.18, 96.8),
    EEG_CHANNEL('TP8',118, 0.523, -45.5, -85.5, -7.13, -118, -4.21, 97.2),
    EEG_CHANNEL('PO7',-151, 0.492, -97.5, 54.8, 2.79, 151, 1.43, 112),
    EEG_CHANNEL('PO8',150, 0.492, -97.6, -55.7, 2.73, -150, 1.39, 112),
    EEG_CHANNEL('OZ',180, 0.460, -115, -0.108, 14.7, -180, 7.27, 116)
    ])

    # Adding CPz
    cp1 = [ch for ch in chanlocs if ch.label == 'CP1'][0]
    cp2 = [ch for ch in chanlocs if ch.label == 'CP2'][0]
    cz = [ch for ch in chanlocs if ch.label == 'CZ'][0]
    pz = [ch for ch in chanlocs if ch.label == 'PZ'][0]

    cpz_x = np.mean([cp1.x, cp2.x, cz.x, pz.x])
    cpz_y = np.mean([cp1.y, cp2.y, cz.y, pz.y])
    cpz_z = np.mean([cp1.z, cp2.z, cz.z, pz.z])

    cpz_sph_theta, cpz_sph_phi, cpz_sph_r = cart2sph(cpz_x, cpz_y, cpz_z)
    cpz_sph_theta = np.degrees(cpz_sph_theta)
    cpz_sph_phi = np.degrees(cpz_sph_phi)

    cpz_theta = 180  # all in the same line are 180
    cpz_radius = (0.247 + 0.029) / 2  # mean between Cz and Pz

    chanlocs = np.append(chanlocs, EEG_CHANNEL('CPZ',cpz_theta, cpz_radius, cpz_x, cpz_y, cpz_z, cpz_sph_theta, cpz_sph_phi, cpz_sph_r))

    return chanlocs



def get_laplacianMask(channels=None, distance=None, isDistanceMeasure=False, chanlocs=get_standard_chanlocs()):
    """
    Returns the classic laplacian mask, weighted on the distance and the chanlocs.
    
    Parameters:
    - channels: list of channel name strings. The lapMask is based on this channels order
    - distance: int of neighbours (default:all) or float cutoff distance (default:1) to take into account.
                The channels distance is normalized to 1 (between FPz and Oz or the greatest distance 
                between channels if another chanlocs is given)
    - isDistanceMeasure: bool, flag to select if variable distance is the number of neighbours (default) 
                         or the cutoff distance
    - chanlocs: custom array of EEG_CHANNELS
    
    Returns:
    - lapMask: numpy array, classic laplacian mask
    - weighted_lapMask: numpy array, weighted laplacian mask
    - chanlocs: updated chanlocs
    """

    if channels is None:
        channels = [ch.label for ch in chanlocs]
    else:
        channels = [ch.upper() for ch in channels]
    
    if distance is None:
        distance = len(channels) - 1

    labels, ch_idx, locs_idx = np.intersect1d(channels, [ch.label for ch in chanlocs], return_indices=True)
    
    if len(labels) != len(channels):
        not_found = set(channels) - set(labels)
        print(f"!!!!   {len(channels) - len(labels)} channel(s) not found in chanlocs. Not found: {', '.join(not_found)}   !!!!")

    # Get distance between channels + normalization + only wanted channels
    pos_chanlocs = np.array([[ch.x, ch.y, ch.z] for ch in chanlocs])
    locs_dist = squareform(pdist(pos_chanlocs))
    locs_dist /= np.max(locs_dist)
    locs_dist = locs_dist[np.ix_(locs_idx, locs_idx)]

    # Reordering the locs based on channels
    chan_dist = np.zeros((len(channels), len(channels)))
    chan_dist[np.ix_(ch_idx, ch_idx)] = locs_dist

    # Removing channels that do not meet requirements
    if isDistanceMeasure:
        chan_dist[chan_dist > distance] = 0
    else:
        n_max = len(channels) - (distance + 1)  # Take into account channels itself
        if n_max < 0:
            print(f"!!!!   Cannot take into account {distance} neighbours because there are {len(channels)} channels. Taking into account all channels.   !!!!")
        elif n_max > 0:
            idx_max = np.argsort(chan_dist, axis=1)
            for n_chan in range(len(channels)):
                chan_dist[idx_max[n_chan,-n_max:], n_chan] = 0
            

    weighted_lapMask = np.zeros(chan_dist.shape)
    lapMask = np.zeros(chan_dist.shape)

    # Normalize matrix where diag=1 and sum column = 0
    for n_chan in range(len(channels)):
        col = chan_dist[:, n_chan]
        col[col > 0] = 1 / col[col > 0]
        weighted_lapMask[:, n_chan] = -col / np.sum(col)
        weighted_lapMask[n_chan, n_chan] = 1

        col[col > 0] = -1 / np.sum(col > 0)
        lapMask[:, n_chan] = col
        lapMask[n_chan, n_chan] = 1


    chanlocs = chanlocs[locs_idx]
    new_chanlocs = chanlocs.copy()
    new_chanlocs[ch_idx] = chanlocs

    return lapMask, weighted_lapMask, new_chanlocs



def plot_lapMask(lapmask, chanlocs):

    y = [chan.x for chan in chanlocs]
    x = [chan.y for chan in chanlocs]
    z = [chan.z for chan in chanlocs]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    
    for i, chan in enumerate(chanlocs):
        ax.text(x[i], y[i], z[i], chan.label)
    
    cs, cd = np.where(lapmask - np.eye(len(lapmask)) != 0)
    for i in range(len(cs)):
        ax.plot([x[cs[i]], x[cd[i]]], [y[cs[i]], y[cd[i]]], [z[cs[i]], z[cd[i]]])
    
    plt.show()

