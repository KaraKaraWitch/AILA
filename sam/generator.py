import os
import numpy as np
from sam.config import (
    nb_gaussian,
    shape_r_gt,
    shape_c_gt,
    shape_r,
    shape_c,
    shape_r_out,
    shape_c_out,
)
from sam.utilities import (
    preprocess_images,
    preprocess_maps,
    preprocess_fixmaps,
)


def generator(
    b_s,
    imgs_path,
    maps_path,
    fixs_path,
):
    images = [
        os.path.join(imgs_path, fname)
        for fname in os.listdir(imgs_path)
        if fname.endswith((".jpg", ".jpeg", ".png"))
    ]
    maps = [
        os.path.join(maps_path, fname)
        for fname in os.listdir(maps_path)
        if fname.endswith((".jpg", ".jpeg", ".png"))
    ]
    fixs = [
        os.path.join(fixs_path, fname)
        for fname in os.listdir(fixs_path)
        if fname.endswith(".mat")
    ]

    images.sort()
    maps.sort()
    fixs.sort()

    n_images = len(images)

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    img_yielded = 0
    while True:
        X = (preprocess_images(images[counter : counter + b_s], shape_r, shape_c),)
        Y = preprocess_maps(maps[counter : counter + b_s], shape_r_out, shape_c_out)
        Y_fix = preprocess_fixmaps(
            fixs[counter : counter + b_s], shape_r_out, shape_c_out
        )
        yield [X, gaussian], [Y, Y, Y_fix]

        img_yielded += 1
        if img_yielded == n_images:
            break
        else:
            counter = (counter + b_s) % n_images


def generator_test(b_s, imgs_test_path):
    images = [
        os.path.abspath(os.path.join(imgs_test_path, fname))
        for fname in os.listdir(imgs_test_path)
        if fname.endswith((".jpg", ".jpeg", ".png"))
    ]
    images.sort()
    n_images = len(images)
    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    img_yielded = 0
    while True:
        yield [
            [
                preprocess_images(images[counter : counter + b_s], shape_r, shape_c),
                gaussian,
            ]
        ]
        img_yielded += 1
        if img_yielded == n_images:
            break
        else:
            counter = (counter + b_s) % n_images
