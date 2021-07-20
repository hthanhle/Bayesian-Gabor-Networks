import numpy as np
# import matplotlib.pyplot as plt
import cv2


# In[1]: Overlay a BINARY groundtruth with a input image
def visualize(groundtruth, image):  # 'groundtruth'is a 2D array (i.e. grayscale), 'image' is a normal RGB image
    background = [0, 0, 0]
    lane = [0, 255, 0]  # Lane = green color
    num_classes = 2

    gt_colours = np.array([background, lane])  # expcted shape: (2, 3)
    r = groundtruth.copy()
    g = groundtruth.copy()
    b = groundtruth.copy()

    for c in range(num_classes):  # 0: background, 1: lane
        r[groundtruth == c] = gt_colours[c, 0]
        g[groundtruth == c] = gt_colours[c, 1]
        b[groundtruth == c] = gt_colours[c, 2]

    # Fuse 3 color channels    
    rgb = np.zeros((groundtruth.shape[0], groundtruth.shape[1], 3))
    rgb[:, :, 0] = (r / 255.0)
    rgb[:, :, 1] = (g / 255.0)
    rgb[:, :, 2] = (b / 255.0)

    rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    overlaid_img = cv2.addWeighted(image, 1, rgb, 0.3, 0)
    return overlaid_img


# In[1]: Overlay a 3-class groundtruth with a input image
def visualize_3_classes(groundtruth,
                        image):  # 'groundtruth'is a 2D array (i.e. grayscale), 'image' is a normal RGB image
    background = [0, 0, 0]
    lane = [0, 255, 0]  # Lane = green color
    marker = [255, 0, 0]  # Marker = red color
    num_classes = 3

    gt_colours = np.array([background, lane, marker])
    r = groundtruth.copy()
    g = groundtruth.copy()
    b = groundtruth.copy()

    for c in range(num_classes):  # 0: background, 1: lane, 2: marker
        r[groundtruth == c] = gt_colours[c, 0]
        g[groundtruth == c] = gt_colours[c, 1]
        b[groundtruth == c] = gt_colours[c, 2]

    # Fuse 3 color channels    
    rgb = np.zeros((groundtruth.shape[0], groundtruth.shape[1], 3))
    rgb[:, :, 0] = (r / 255.0)
    rgb[:, :, 1] = (g / 255.0)
    rgb[:, :, 2] = (b / 255.0)

    rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    overlaid_img = cv2.addWeighted(image, 1, rgb, 0.9, 0)
    return overlaid_img
