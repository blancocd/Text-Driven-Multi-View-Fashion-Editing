import numpy as np
import cv2

fourddress_palette = np.array([
    [128., 128., 128.],   # 0 skin
    [255., 128.,   0.],   # 1 hair
    [128.,   0., 255.],   # 2 shoes
    [255.,   0.,   0.],   # 3 inner
    [  0., 255.,   0.],   # 4 lower
    [  0., 128., 255.],   # 5 outer
])

fourddress_labels = ['skin', 'hair', 'shoes', 'inner', 'lower', 'outer']

def remove_unconn(mask, min_component_area = 20):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            filtered_mask[labels == i] = 255
    
    mask = filtered_mask 
    return mask

def get_mask_from_segmap(segmentation_map, target_colors, avoid_colors, tolerance=5, dil_its=1, ero_its = 1) -> np.ndarray:
    mask = np.zeros(segmentation_map.shape[:2], dtype=np.uint8)

    for target in target_colors:
        close = np.all(np.abs(segmentation_map - target) <= tolerance, axis=-1)
        mask = np.logical_or(mask, close)

    mask = (mask * 255).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=dil_its)

    # include for removing body from mask
    body_mask = np.zeros(segmentation_map.shape[:2], dtype=np.uint8)
    for avoid in avoid_colors:
        close = np.all(np.abs(segmentation_map - avoid) <= tolerance, axis=-1)
        body_mask = np.logical_or(body_mask, close)
    body_mask = (body_mask * 255).astype(np.uint8)
    body_mask = cv2.erode(body_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=ero_its)
    mask = np.logical_and(mask, ~body_mask)
    return mask

def get_mask_4ddress(segmentation_map, seg_label, dil_its=1, ero_its=1) -> np.ndarray:
    target_colors = set()
    if 'skin' in seg_label:
        target_colors.add(0)
    elif 'inner' in seg_label:
        target_colors.add(3)
    elif 'hair' in seg_label:
        target_colors.add(1)
    elif 'lower' in seg_label:
        target_colors.add(4)
    elif 'outer' in seg_label:
        target_colors.add(5)
    elif 'upper' in seg_label:
        target_colors.update([3, 5])
    elif 'human' in seg_label:
        target_colors.update(list(range(6)))
    target_colors = [fourddress_palette[i] for i in target_colors]
    if ero_its is -1:
        avoid_colors = []
    else:
        avoid_colors = [fourddress_palette[i] for i in [0, 1]] # skin and hair
    mask = get_mask_from_segmap(segmentation_map, target_colors, avoid_colors, dil_its=dil_its, ero_its=ero_its)
    return mask
