import pathlib
import cv2
import os

import tensorflow as tf
from keras.optimizers import RMSprop

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.models import Model
from keras.utils import get_file

from .utilities import postprocess_predictions
from .models import (
    kl_divergence,
    correlation_coefficient,
    nss,
    sam_resnet,
)
from .generator import generator, generator_test, generator_image
from .cropping import batch_crop_images, script_crop
from PIL import Image
from .config import (
    shape_r,
    shape_c,
    shape_r_gt,
    shape_c_gt,
    nb_gaussian,
    b_s,
    nb_epoch,
    steps_per_epoch,
    aspect_ratio,
    retained_attention,
    DATASET_IMAGES_URL,
    DATASET_MAPS_URL,
    DATASET_FIXS_URL,
    SAM_RESNET_SALICON_2017_WEIGHTS,
    dataset_path,
    gaussina_sigma,
)


tf.keras.backend.set_image_data_format(data_format="channels_first")


class SalMap:
    @classmethod
    def auto(cls):
        salmap = cls()
        salmap.compile()
        salmap.load_weights()
        salmap.predict_maps()
        salmap.box_and_crop()

    def __init__(self):
        self.x = Input((3, shape_r, shape_c))
        self.x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))

    def compile(self):
        self.model = Model(
            inputs=[self.x, self.x_maps], outputs=sam_resnet([self.x, self.x_maps])
        )
        self.model.compile(
            RMSprop(learning_rate=1e-4),
            loss=[kl_divergence, correlation_coefficient, nss],
            loss_weights=[10, -2, -1],
        )

    @staticmethod
    def get_valid_images(imgs_path, maps_path, fixs_path):
        _images = {
            fname.rsplit(".", 1)[0]: os.path.join(imgs_path, fname)
            for fname in os.listdir(imgs_path)
            if fname.endswith((".jpg", ".jpeg", ".png"))
        }
        _maps = {
            fname.rsplit(".", 1)[0]: os.path.join(maps_path, fname)
            for fname in os.listdir(maps_path)
            if fname.endswith((".jpg", ".jpeg", ".png"))
        }

        _fixs = {
            fname.rsplit(".", 1)[0]: os.path.join(fixs_path, fname)
            for fname in os.listdir(fixs_path)
            if fname.endswith(".mat")
        }

        images = []
        maps = []
        fixs = []

        # make sure all files in images have corresponding files in maps and fixs
        for fname in set(_images).intersection(_maps, _fixs):
            images.append(_images[fname])
            maps.append(_maps[fname])
            fixs.append(_fixs[fname])

        del _images, _maps, _fixs

        return images, maps, fixs

    def load_weights(self, weights_dir=None):
        if weights_dir:
            if not os.path.exists(weights_dir):
                weights_dir = None
        else:
            fname = "sam-resnet_salicon_weights.pkl"
            cache_subdir = "weights"

            weights_dir = get_file(
                fname,
                SAM_RESNET_SALICON_2017_WEIGHTS,
                cache_subdir=cache_subdir,
                file_hash="92b5f89fd34a3968776a5c4327efb32c",
            )

        self.model.load_weights(weights_dir)

    def predict_maps(self, imgs_test_path="/samples", sigma=gaussina_sigma):
        if imgs_test_path.startswith("/"):
            imgs_test_path = imgs_test_path.rsplit("/", 1)[1]
        # Output Folder Path
        if os.path.exists(imgs_test_path):
            self.imgs_test_path = imgs_test_path
        else:
            self.imgs_test_path = os.path.join(os.getcwd(), imgs_test_path)
        if not os.path.exists(self.imgs_test_path):
            raise Exception(
                f"Couldn't find the directory {imgs_test_path} or {self.imgs_test_path}"
            )

        home_dir = os.path.dirname(self.imgs_test_path)
        maps_folder = os.path.join(home_dir, "maps")
        if not os.path.exists(maps_folder):
            os.mkdir(maps_folder)
        cmaps_folder = os.path.join(home_dir, "cmaps")
        if not os.path.exists(cmaps_folder):
            os.mkdir(cmaps_folder)

        file_names = [
            fname
            for fname in os.listdir(self.imgs_test_path)
            if fname.endswith((".jpg", ".jpeg", ".png"))
        ]
        file_names.sort()

        print("Predicting saliency maps for " + self.imgs_test_path)
        predictions = self.model.predict(
            generator_test(file_names, imgs_test_path), steps=len(file_names)
        )[0]

        for pred, fname in zip(predictions, file_names):
            image_path = os.path.join(self.imgs_test_path, fname)
            original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mapped_image = postprocess_predictions(
                pred[0], original_image.shape[0], original_image.shape[1]
            )
            map_path = os.path.join(maps_folder, fname)
            cv2.imwrite(map_path, mapped_image)

            gray_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            colored_map = cv2.applyColorMap(gray_map, cv2.COLORMAP_JET)
            overlaid = cv2.addWeighted(original_image, 0.5, colored_map, 0.5, 0)
            cmap_path = os.path.join(cmaps_folder, fname)
            cv2.imwrite(cmap_path, overlaid)

    def shinon_predicts(
        self, images: list[pathlib.Path], gaussian_blur=1, visualize=False, crop_args={}
    ):
        for image in images:
            with Image.open(image) as pil_image:
                if pil_image.size[0] > 768 or pil_image.size[1] > 768:
                    # Resize image to 768 by 768 at best or else the processing is gonna take forever.
                    sized = pil_image.copy()
                    sized.thumbnail((768, 768))
                    if sized.size[0] == 768:
                        ratio = pil_image.size[0] / sized.size[0]
                    elif sized.size[1] == 768:
                        ratio = pil_image.size[1] / sized.size[1]
                    else:
                        raise Exception("Uncaught Error on ratios")
                else:
                    sized = pil_image
                    ratio = 1.0

                predictions = self.model.predict(generator_image(sized), steps=1)[0][0][
                    0
                ]
                mapped_image = postprocess_predictions(
                    predictions, sized.size[1], sized.size[0], sigma=gaussian_blur
                )
                salient_image = Image.fromarray(mapped_image)
                # sized.show()
                # salient_image.show()
                # if visualize:
                #    salient_image.show()

                if (
                    "dsc_asp" in crop_args
                    and isinstance(crop_args["dsc_asp"], str)
                    and crop_args["dsc_asp"].lower() == "auto"
                ):
                    crop_args["dsc_asp"] = pil_image.size[0] / pil_image.size[1]

                if crop_args:
                    _, coords_scaled, visualize = script_crop(
                        salient_image,
                        ratio,
                        visualize=None if not visualize else sized,
                        **crop_args,
                    )
                else:
                    _, coords_scaled, visualize = script_crop(
                        salient_image, ratio, visualize=None if not visualize else sized
                    )

                for idx, coord in enumerate(coords_scaled):
                    image_path = pathlib.Path(
                        f"output/{image.with_stem(image.stem + '_' + str(idx)).name}"
                    ).resolve()
                    if not image_path.parent.is_dir():
                        image_path.parent.mkdir(parents=True)
                    if image.suffix in [".jpg", ".jpeg"]:
                        pil_image.crop(coord).save(image_path, quality=95)
                    else:
                        pil_image.crop(coord).save(image_path)
                if visualize:
                    image_path = pathlib.Path(
                        f"output/{image.with_stem(image.stem + '_vis').with_suffix('.png').name}"
                    ).resolve()
                    visualize.save(image_path)

    @staticmethod
    def box_and_crop(
        originals_folder="samples",
        max_gap=0.2,
        peak_thr=0.5,
    ):
        """
        get_centroids's maximum_gap & peak_threshold
        """
        maps_folder = os.path.join(os.path.dirname(originals_folder), "maps")
        if not os.path.exists(maps_folder):
            raise Exception(
                f"Saliency mappings for the images in {originals_folder}"
                " must be present in {maps_folder}.\n"
                "Run this command - salmap.test(imgs_test_path=<original-images>) and try again!"
            )
        crops_folder = os.path.join(os.path.dirname(originals_folder), "crops")
        boxes_folder = os.path.join(os.path.dirname(originals_folder), "boxes")

        batch_crop_images(
            originals_folder,
            maps_folder,
            crops_folder,
            boxes_folder,
            max_gap=max_gap,
            peak_thr=peak_thr,
        )
