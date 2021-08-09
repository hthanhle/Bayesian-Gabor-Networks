import numpy as np
import cv2


def visualize(seg_map, img):
    """
    Overlay a segmentation map with an input color image
    :param seg_map: segmentation map of size H x W
    :param img: original image of size H x W x 3
    :return: overlaid image
    """
    # Class 0 (Background): Black
    # Class 1 (Lane): Green
    COLOR_CODE = [[0, 0, 0], [0, 1, 0]]
    seg_map_rgb = np.zeros(img.shape)

    # Convert the segmentation map (with class IDs) to a RGB image
    for k in np.unique(seg_map):
        seg_map_rgb[seg_map == k] = COLOR_CODE[k]
    seg_map_rgb = (seg_map_rgb * 255).astype('uint8')

    # Super-impose the color segmentation map onto the original image
    overlaid_img = cv2.addWeighted(img, 1, seg_map_rgb, 0.9, 0)

    return overlaid_img
