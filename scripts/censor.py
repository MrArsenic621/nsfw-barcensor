import random
from unittest import result

import torch
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from PIL import Image

from torchvision.utils import save_image,make_grid
from torchvision.io import read_image
import torchvision

from modules import scripts, shared

# *****************************
from nudenet import NudeDetector

detector = NudeDetector()

import cv2
import numpy as np
import os

from random import randint

import torchvision.transforms as transforms

parts = [
    "EXPOSED_ANUS",
    "EXPOSED_ARMPITS",
    "COVERED_BELLY",
    "EXPOSED_BELLY",
    "COVERED_BUTTOCKS",
    "EXPOSED_BUTTOCKS",
    "FACE_F",
    "FACE_M",
    "COVERED_FEET",
    "EXPOSED_FEET",
    "COVERED_BREAST_F",
    "EXPOSED_BREAST_F",
    "COVERED_GENITALIA_F",
    "EXPOSED_GENITALIA_F",
    "EXPOSED_BREAST_M",
    "EXPOSED_GENITALIA_M",
]

parts_to_censor = [
    "EXPOSED_ANUS",
    "EXPOSED_BREAST_F",
    "EXPOSED_GENITALIA_F",
    "EXPOSED_GENITALIA_M",
]


# *****************************


def censor_batch(x):
    res = []
    
    #from PIL import Image
    
    
    
    for i, tensor in enumerate(x):
        img_path = f"./extensions/nsfw-barcensor/scripts/tmp/{i}.png"
        save_image(tensor, img_path)
        #im_save(tensor, img_path)

    for j in range(i + 1):
        img_path = f"./extensions/nsfw-barcensor/scripts/tmp/{j}.png"
        out_path = f"./extensions/nsfw-barcensor/scripts/out/{j}.png"

        detector.censor(img_path=img_path,
                        out_path=out_path,
                        parts_to_blur=parts_to_censor,
                        )

        # img = cv2.cvtColor(cv2.imread(out_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        img = cv2.imread(out_path)
        # img = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = img/255.0
        #img = img.astype(np.float32)
        #img = cv2.bitwise_not(img)
        
        
        res.append(img)

        # image = cv2.imread('/data/cat.jpg', cv2.IMREAD_UNCHANGED)
        os.remove(img_path)
        os.remove(out_path)

    y_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()

    x_samples_ddim_numpy = np.array(res).astype(np.float32) / 255.0
    x = torch.from_numpy(x_samples_ddim_numpy).permute(0, 3, 1, 2)

    # x = x.byte()

    print("checked for nsfw content\n")

    # print("\n", x_samples_ddim_numpy.dtype)
    # print("\n", y_samples_ddim_numpy.dtype)
    # print("\n", np.array_equiv(x_samples_ddim_numpy, y_samples_ddim_numpy))

    # for tensor in x:
    #     img_path = f"./extensions/nsfw-barcensor/scripts/test/{random.randint(1,100000)}.png"
    #     save_image(tensor, img_path)

    return x


class NsfwCheckScript(scripts.Script):
    def title(self):
        return "NSFW check bar"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs['images']
        images[:] = censor_batch(images)[:]
