import os
from pathlib import Path

import cv2
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import xmltodict
from PIL import Image
import tifffile

Image.MAX_IMAGE_PIXELS = None


def create_mask_from_vpa(image_path, vpa_path) -> Image:
    image = Image.open(image_path)
    w, h = image.size
    mask_array = np.zeros((h, w), np.uint8)

    with open(vpa_path, 'r') as file:
        polygon_data = file.read()

    polygon_dict = xmltodict.parse(polygon_data)
    if polygon_dict['Annotations']['Polylines'] is not None:
        polylines = polygon_dict['Annotations']['Polylines']['Polyline']
    else:
        polylines = polygon_dict['Annotations']['Markers']['Marker']

    if not isinstance(polylines, list):
        polylines = [polylines]
    for polyline in polylines:
        points = polyline['Points']
        points = list(map(lambda x_y: list(map(float, x_y.split(','))), points.split(' ')))
        points = np.array(points, dtype='int32')
        points = points[:, :-1]  # dropping error values
        cv2.fillPoly(mask_array, pts=np.array([points], dtype='int32'), color=1)

    return Image.fromarray(mask_array * 255, mode='L').convert('1')


def create_empty_mask(image_path):
    image = Image.open(image_path)
    w, h = image.size
    mask_array = np.zeros((h, w), np.uint8)
    return Image.fromarray(mask_array, mode='L').convert('1')


def convert_tif_images_to_jpg(directory, remove_tifs=True):
    if not isinstance(directory, Path):
        directory = Path(directory)
    tiffs = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.tif') or file_name.endswith('.tiff'):
            tiffs.append(file_name)

            image_path = directory / file_name
            img_array = tifffile.imread(image_path)
            # img = Image.fromarray(img_array).convert('RGB')
            # img.save(directory / f'{image_path.stem}.jpg')
            matplotlib.image.imsave(directory / f'{image_path.stem}.jpg', img_array)
            os.remove(image_path)


if __name__ == '__main__':
    image_name = '96-6433-2-20X-0.50 ph1__UPlanF1__Olympu--L-U-'
    mask = create_mask_from_vpa(f'{image_name}.tif', f'{image_name}.vpa')
    print(np.prod(np.array(mask).shape))
    print(np.array(mask).sum())
    plt.imshow(mask, cmap='gray')
    plt.show()
    mask.save(f'{image_name}-mask.jpg')
