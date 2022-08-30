import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
import shutil


def get_rectangle_params_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height


def draw_bboxes(plot_ax, bboxes, class_labels, get_rectangle_corners_fn=get_rectangle_params_from_pascal_bbox):
    for bbox, label in zip(bboxes, class_labels):
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left, width, height, linewidth=4, edgecolor="black", fill=False,
        )
        rect_2 = patches.Rectangle(
            bottom_left, width, height, linewidth=2, edgecolor="white", fill=False,
        )
        rx, ry = rect_1.get_xy()

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)
        plot_ax.annotate(label, (rx + width, ry + height), color='white', fontsize=20)


def show_image(image, bboxes=None, class_labels=None, draw_bboxes_fn=draw_bboxes):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    if bboxes:
        if class_labels is None:
            class_labels = [''] * len(bboxes)
        draw_bboxes_fn(ax, bboxes, class_labels)

    plt.show()


# This function flattens a directory with complex tree of subdirectories into a single directory
def flatten_folder(folder_path, remove_dirs=True):
    # convert folder_path to Path if it isn't already
    if not isinstance(folder_path, Path):
        folder_path = Path(folder_path)
    # gather all path of all files in this directory tree
    all_files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if
                 Path(root) != folder_path]
    # move files to top level directory
    for file in all_files:
        shutil.move(file, folder_path)
    # remove redundant subdirectories if remove_dirs is True:
    if remove_dirs:
        for subdir in [os.path.join(folder_path, directory) for directory in next(os.walk(folder_path))[1]]:
            shutil.rmtree(subdir)
