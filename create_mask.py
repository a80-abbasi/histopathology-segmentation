import cv2
import matplotlib.pyplot as plt
import numpy as np
import xmltodict
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def create_mask_from_vpa(image_path, vpa_path):
    image = Image.open(image_path)
    w, h = image.size
    mask_array = np.ones((h, w), np.uint8)

    with open(vpa_path, 'r') as file:
        polygon_data = file.read()

    polygon_dict = xmltodict.parse(polygon_data)
    polylines = polygon_dict['Annotations']['Polylines']['Polyline']
    if not isinstance(polylines, list):
        polylines = [polylines]
    for polyline in polylines:
        points = polyline['Points']
        points = list(map(lambda x_y: list(map(float, x_y.split(','))), points.split(' ')))
        points = np.array(points, dtype='int32')
        points = points[:, :-1]  # dropping error values
        cv2.fillPoly(mask_array, pts=np.array([points], dtype='int32'), color=0)

    return image, Image.fromarray(mask_array)


image, mask = create_mask_from_vpa('test_image.jpg', 'test_mask.vpa')
plt.imshow(mask, cmap='gray')
plt.show()

# image = Image.open('test_image.tif')
plt.imshow(image)
plt.show()
