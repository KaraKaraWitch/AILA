import os
import typing
from PIL import Image
import numpy as np
import cv2
import math
from skimage.feature import peak_local_max
from scipy.cluster.vq import kmeans


def find_max_subarray(array: np.ndarray, window_w: int, threshold: float) -> tuple:
    assert len(array.shape) == 1

    best_sum = -1
    start_idx = None

    array_cum = np.pad(np.cumsum(array), pad_width=[(1, 0)])

    max_start_idx = array.shape[0] - window_w

    for idx in range(max_start_idx + 1):
        cumsum_upto_windowend = array_cum[idx + window_w]
        cumsum_before_windowstart = array_cum[idx]
        subarray_sum = cumsum_upto_windowend - cumsum_before_windowstart
        if subarray_sum > threshold and subarray_sum > best_sum:
            best_sum = subarray_sum
            start_idx = idx

    return start_idx, best_sum


def find_rectangle(img_array, asp_ratio, keep_attention):
    img_h, img_w = img_array.shape
    if img_h > img_w:
        transpose = True
        array2d = img_array.T
        img_h, img_w = img_w, img_h
    else:
        transpose = False
        array2d = img_array
    array2d_hcum = np.pad(np.cumsum(array2d, axis=0), pad_width=[(1, 0), (0, 0)])
    img_h, img_w = array2d.shape
    img_area = img_h * img_w

    total_attention = np.sum(array2d)
    threshold_attention = keep_attention * total_attention

    # initialize
    y_start = 0
    min_height = 1
    y_finish = y_start + min_height
    best_area = img_area
    failedToFind = True

    while True:
        window_h = y_finish - y_start
        window_w = math.ceil(asp_ratio * window_h)

        if not (
            y_finish <= img_h
            and window_h >= min_height
            and window_h <= img_h
            and window_w <= img_w
        ):
            break

        subarray2d = array2d_hcum[y_finish] - array2d_hcum[y_start]
        x_start, attention_kept = find_max_subarray(
            subarray2d, window_w, threshold_attention
        )
        if attention_kept > 0:
            box_area = window_w * window_h
            if (box_area < best_area) or (
                box_area == best_area and attention_kept > best_attention
            ):
                best_area = box_area
                x, y, w, h = x_start, y_start, window_w, window_h
                best_attention = attention_kept
                failedToFind = False
            y_start += 1
        else:
            y_finish += 1

    if failedToFind:
        return {}
    else:
        attention_factor = best_attention / total_attention
        area_factor = w * h / img_area
        density_factor = attention_factor / area_factor
        return {
            "coords": (y, x, h, w) if transpose else (x, y, w, h),
            "area": area_factor,
            "attention": attention_factor,
            "density": density_factor,
        }


def find_best_rectangle(
    salient_ndimage, a_r, min_attention, step=0.02, alpha=10, beta=10, gamma=0.1
):
    results = {}
    attention = 1
    count = 0
    while attention >= min_attention:
        attention -= step * (2**count)
        count += 1
        result = find_rectangle(salient_ndimage, a_r, attention)
        if result:
            score = (
                -alpha * math.log10(1 - result["attention"])
                - beta * math.log10(1 - result["area"])
                + gamma ** (result["density"])
            )
            results[score] = result.pop("coords")

    if results:
        sorted_scores = sorted(results)
        print(sorted_scores)
        x, y, w, h = results[sorted_scores[-1]]
        return x, y, w, h
    else:
        raise Exception(
            f"Failed to crop with aspect ratio: {a_r}, minimum attention: {min_attention}"
        )


def get_centroids(array2d, maximum_gap=0.2, peak_theshold=0.5, reverse_k=False, max_k=4):
    array2d = array2d.copy()
    #print(array2d.shape)

    centroid_k = [i+1 for i in range(max_k)]
    if reverse_k:
        centroid_k.reverse()
    maximum_distortion = array2d.shape[0] * maximum_gap
    for _k in centroid_k:
        peaks = peak_local_max(array2d, threshold_rel=peak_theshold).astype(np.float32)
        k_peaks, distortion = kmeans(peaks.astype(float), _k)
        if distortion < maximum_distortion:
            return k_peaks.astype(np.uint32)
    return []


def descend_from_hilltop(array2d, cent_ij, alpha=1.5, beta=0.5, asp_ratio=1.44):
    cent_i, cent_j = cent_ij
    print("hilltop:",array2d.shape, cent_ij)
    image_h, image_w = array2d.shape
    _1_pct_height = int(image_h * 0.05)
    total_area = image_h * image_w
    total_attention = array2d.sum()

    scores = []
    attentions = []
    densities = []
    coords = []

    pad_top = _1_pct_height
    pad_bottom = _1_pct_height
    while True:
        pad_right = asp_ratio * pad_bottom
        pad_left = asp_ratio * pad_top

        start_i = int(cent_i - pad_top)
        start_j = int(cent_j - pad_left)

        finish_i = int(cent_i + pad_bottom)
        finish_j = int(cent_j + pad_right)

        if start_i < 0 or finish_i >= image_h or start_j < 0 or finish_j >= image_w:
            break
        else:
            attention = array2d[start_i:finish_i, start_j:finish_j].sum()
            attention_factor = attention / total_attention
            attentions.append(attention_factor)

            area = (finish_i - start_i + 1) * (finish_j - start_j + 1)
            area_factor = area / total_area

            density_factor = attention_factor / area_factor
            densities.append(density_factor)

            coords.append([start_i, start_j, finish_i, finish_j])

            pad_bottom += _1_pct_height
            pad_top += _1_pct_height

    attentions = np.array(attentions)
    densities = np.array(densities)
    scores = np.tanh(densities**alpha) * (attentions**beta)

    start_i, start_j, finish_i, finish_j = coords[np.argmax(scores)]
    start_x, start_y, finish_x, finish_y = start_j, start_i, finish_j, finish_i

    return start_x, start_y, finish_x, finish_y


def crop(
    original_image_path,
    saliency_map_path,
    cropped_image_path,
    boxed_image_path,
    max_gap=0.2,
    peak_thr=0.5,

):
    BLUE = (255, 0, 0)
    THICKNESS = 5

    original_ndimage = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    boxed = np.copy(original_ndimage)
    salient_ndimage = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)

    for i, cent_ij in enumerate(get_centroids(salient_ndimage, maximum_gap=max_gap, peak_theshold=peak_thr)):
        if i > 0:
            name, ext = cropped_image_path.rsplit(".", 1)
            cropped_image_path = f"{name}_{i}.{ext}"

        start_x, start_y, finish_x, finish_y = descend_from_hilltop(
            salient_ndimage, cent_ij
        )
        cropped_ndimage = original_ndimage[start_y:finish_y, start_x:finish_x, :]
        cv2.imwrite(cropped_image_path, cropped_ndimage)
        cv2.rectangle(boxed, (start_x, start_y), (finish_x, finish_y), BLUE, THICKNESS)
    cv2.imwrite(boxed_image_path, boxed)

def script_crop(salient_map:Image.Image,
                point_scale: float,
                visualize:typing.Optional[Image.Image]=None,
                max_gap=0.2, peak_thr=0.5,
                dsc_alpha=1.5, dsc_beta=0.5, dsc_asp=1.44,
                reverse_k=False, max_k=4
                ):
    """Gets the coordinates for the cropping region

    Args:
        salient_map (Image.Image): The salient image map. (Required)
        point_scale (float): The point scaling to map the salient image to the original image
        max_gap (float, optional): Determines how wide in terms of spread the saliency will need before it is picked up as the center. Defaults to 0.2.
        peak_thr (float, optional): Determines how high the peak for a "hotspot"/"centroid" needs to be before it gets picked up. Defaults to 0.5.
        dsc_alpha (float, optional): Hilltop Descend's alpha param. Defaults to 1.5.
        dsc_beta (float, optional): Hilltop Descend's Beta param. Defaults to 0.5.
        dsc_asp (float, optional): Hilltop Descend's Aspect Ratio. Defaults to 1.44.
        visualize (Image.Image, optional): pass in the scaled image to visualize how the points are drawn.
    Returns:
        A tuple containing a list of coordinates and another list with their values scaled.
    """
    
    salient_ndimage = np.array(salient_map, dtype=np.float64)[:].copy()
    coords = []
    coords_scaled = []
    if visualize:
        from PIL import ImageDraw, ImageOps
        salient_map = salient_map.convert("L")
        salient_map = ImageOps.colorize(salient_map, black ="blue", white ="red")
        salient_map = salient_map.convert("RGBA")
        salient_map.putalpha(127)
        visualize = visualize.convert("RGBA")
        visualize.alpha_composite(salient_map)
        salient_draw = ImageDraw.Draw(visualize)
    else:
        salient_draw = None
    
    for i, cent_ij in enumerate(get_centroids(salient_ndimage, maximum_gap=max_gap, peak_theshold=peak_thr, reverse_k=reverse_k, max_k=max_k)):
        x0, y0, x1, y1 = descend_from_hilltop(
            salient_ndimage, cent_ij, dsc_alpha, dsc_beta, dsc_asp
        )
        coords.append([x0, y0, x1, y1])
        coords_scaled.append([x0*point_scale, y0*point_scale, x1*point_scale, y1*point_scale])
        if visualize and salient_draw:
            dot_size = 5
            salient_draw.ellipse((cent_ij[1]-dot_size, cent_ij[0]-dot_size, cent_ij[1]+dot_size, cent_ij[0]+dot_size), fill=(255,0,0))
            print("Centroid",cent_ij, "crop_box", (x0, y0, x1, y1))
            salient_draw.rectangle((x0, y0, x1, y1), outline=(0,0,255))

    #if visualize:
    #    visualize.show()
    return coords, coords_scaled, visualize

    

def batch_crop_images(
    originals_folder,
    maps_folder,
    crops_folder,
    boxes_folder,
    max_gap,
    peak_thr,
):
    for _dir in [crops_folder, boxes_folder]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)

    for fname in os.listdir(maps_folder):
        if fname in os.listdir(originals_folder):
            original_file = os.path.join(originals_folder, fname)
            mapping_file = os.path.join(maps_folder, fname)
            crop_file = os.path.join(crops_folder, fname)
            box_file = os.path.join(boxes_folder, fname)
            try:
                crop_success = crop(
                    original_file,
                    mapping_file,
                    crop_file,
                    box_file,
                    max_gap,
                    peak_thr,
                )
            except Exception as e:
                print(f"{e} for {fname}")
                continue
            else:
                print(f"Cropped and boxed {fname} successfully")
