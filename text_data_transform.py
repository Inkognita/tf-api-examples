import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import scipy.io as sio
import cv2
import io
import PIL
from tqdm import tqdm

import tensorflow as tf
from object_detection.utils import dataset_util

data_root = "/home/liepieshov/dataset/SynthText"
train_tf = os.path.join(data_root, "train.tfrecords")
test_tf = os.path.join(data_root, "test.tfrecords")


class SynthDataSet:
    def __init__(self, data_root, train=None, train_size=0.8):
        self.db = sio.loadmat(os.path.join(data_root, "gt.mat"))
        if train is False:
            self.start = int(self.db['imnames'].shape[1] * 0.8)
        else:
            self.start = 0

    def __getitem__(self, index):
        index += self.start
        fn = self.db['imnames'][0, index][0]
        bbs = self.db['wordBB'][0, index]
        if len(bbs.shape) == 2:
            bbs = np.expand_dims(bbs, -1)
        bbs = np.transpose(bbs, (2, 1, 0))
        return fn, bbs

    def __len__(self):
        return self.db['imnames'].shape[1] - self.start


marginise_to_norm_range = np.vectorize(lambda x: max(min(1.0, x), 0.0))


def create_bound_rect(bb):
    bbox = cv2.boundingRect(bb)
    bbox = np.array(bbox)
    min_max_points = [bbox[:2], bbox[:2] + bbox[2:]]
    min_max_points = np.int0(min_max_points)
    return min_max_points.reshape(-1)


create_bound_rects = lambda bbs: np.array([create_bound_rect(bb) for bb in bbs])


def create_mask(shape, points):
    mask = np.zeros((shape[0], shape[1]))
    cv2.fillConvexPoly(mask, points.astype("int"), 1)
    mask = mask.astype(np.bool)
    return mask.astype(np.float32)


def create_masks(shape, bbs):
    return np.array([create_mask(shape, bb) for bb in bbs])


def create_tf_example(filename, points):
    img_path = os.path.join(data_root, filename)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    width, height = image.size

    bbs = create_bound_rects(points).astype("float")
    bbs[:, ::2] /= width
    bbs[:, 1::2] /= height
    bbs = marginise_to_norm_range(bbs)

    points = points.astype("float")
    mask = create_masks((height, width), points)

    filename = filename.encode("utf-8")  # Filename of the image. Empty if image is not from file
    image_format = b'jpg'  # b'jpeg' or b'png'

    xmins = bbs[:, 0]  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = bbs[:, 2]  # List of normalized right x coordinates in bounding box
    ymins = bbs[:, 1]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = bbs[:, 3]  # List of normalized bottom y coordinates in bounding box

    classes_text = [tf.compat.as_bytes("text")] * bbs.shape[0]  # List of string class name of bounding box (1 per box)
    classes = [1] * bbs.shape[0]  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/mask': (dataset_util.float_list_feature(mask.astype(np.float32).reshape(-1).tolist()))
    }))
    return tf_example


def transform_dataset(dataset):
    train_writer = tf.python_io.TFRecordWriter(train_tf)
    test_writer = tf.python_io.TFRecordWriter(test_tf)

    indexes = list(range(500))
    np.random.shuffle(indexes)

    for ind in tqdm(indexes):
        image, bbs = dataset[ind]

        tf_example = create_tf_example(image, bbs)
        if ind < len(indexes) * 0.8 // 1:
            train_writer.write(tf_example.SerializeToString())
        else:
            test_writer.write(tf_example.SerializeToString())
    train_writer.close()
    test_writer.close()


if __name__ == "__main__":
    dataset = SynthDataSet(data_root=data_root)
    transform_dataset(dataset)
