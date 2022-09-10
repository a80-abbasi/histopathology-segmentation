import os
from pathlib import Path
import tifffile
import matplotlib.pyplot as plt
from PIL import Image

from create_mask import convert_tif_images_to_jpg

Image.MAX_IMAGE_PIXELS = None

# this file is for checking most similar names of masks to missing masks

# masks_path = Path('dataset/masks')
#
# masks = [mask_name for mask_name in os.listdir(masks_path) if mask_name.endswith('.vpa')]
#
# missing_file = '96-6433-1-L-20X-0.50 ph1__UPlanF1__Olympus.vpa'
#
# closest = sorted(masks, key=lambda x: jellyfish.levenshtein_distance(x, missing_file))
# for name in closest:
#     print(name)


# images_path = Path('dataset/images')
# print('96-6433-2-20X-0.50 ph1__UPlanF1__Olympus----R1-.tif' in os.listdir(images_path))
#
# img = tifffile.imread(images_path / '96-6433-2-20X-0.50 ph1__UPlanF1__Olympus----R1-.tif')
# print(img)

convert_tif_images_to_jpg('.')