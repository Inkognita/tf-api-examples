import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import json
import os
from os.path import join as pj

import tensorflow as tf
from object_detection.utils import dataset_util

flags = tf.app.flags
FLAGS = flags.FLAGS

DATA_ROOT = "data"
RAW_IMAGES = pj(DATA_ROOT, "raw-data")
PROCESSED_IMAGES = pj(DATA_ROOT, "img")
TRAIN_TF = pj(DATA_ROOT, "train.tfrecords")
TEST_TF = pj(DATA_ROOT, "test.tfrecords")
RAW_CSV_FILE = pj(DATA_ROOT, "via_region_data.csv")
PROCESSED_CSV_FILE = pj(DATA_ROOT, "via_region_data_processed.csv")

marginise_to_norm_range = np.vectorize(lambda x: max(min(1.0, x), 0.0))


def process_bb(bb):
    bbox = cv2.boundingRect(bb)
    bbox = np.array(bbox)
    min_max_points = [bbox[:2], bbox[:2] + bbox[2:]]
    min_max_points = np.int0(min_max_points)
    return min_max_points.reshape(-1)


def process_row(row):
    img_path = row['filename']
    data_points = json.loads(row['region_shape_attributes'])
    if not data_points:
        raise ValueError("no data")
    bb_x = data_points['all_points_x']
    bb_y = data_points['all_points_y']
    bb = np.array([bb_x, bb_y]).T
    return img_path, process_bb(bb)


def create_tf_example(row):
    img_name, bb = row['filename'], row['bb']
    img_path = os.path.join(PROCESSED_IMAGES, img_name)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    img = cv2.imread(img_path).astype(np.float32)

    bb = eval(",".join(bb.split()))

    height = img.shape[0]  # Image height
    width = img.shape[1]  # Image width

    filename = img_path.encode("utf-8")  # Filename of the image. Empty if image is not from file
    image_format = b'png'  # b'jpeg' or b'png'

    xmins = [bb[2]]  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [bb[0]]  # List of normalized right x coordinates in bounding box
    ymins = [bb[1]]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [bb[3]]  # List of normalized bottom y coordinates in bounding box

    classes_text = [tf.compat.as_bytes("paper")]  # List of string class name of bounding box (1 per box)
    classes = [1]  # List of integer class id of bounding box (1 per box)

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
    }))
    return tf_example


def main(_):
    df = pd.read_csv(PROCESSED_CSV_FILE)

    train_writer = tf.python_io.TFRecordWriter(TRAIN_TF)
    test_writer = tf.python_io.TFRecordWriter(TEST_TF)

    indexes = list(range(len(df)))
    np.random.shuffle(indexes)

    for ind in indexes:
        tf_example = create_tf_example(df.loc[ind])
        if ind < len(indexes) * 0.8 // 1:
            train_writer.write(tf_example.SerializeToString())
        else:
            test_writer.write(tf_example.SerializeToString())

    train_writer.close()
    test_writer.close()


def normalise_images(in_path=RAW_IMAGES, out_path=PROCESSED_IMAGES, in_csv=RAW_CSV_FILE, out_csv=PROCESSED_CSV_FILE):
    df = pd.read_csv(in_csv)
    filenames = []
    bbs = []
    for _, row in df.iterrows():
        try:
            image, bb = process_row(row)
        except ValueError:
            continue

        img_path = pj(in_path, image)
        img = cv2.imread(img_path)
        bb = bb.astype("float")
        bb[::2] = bb[::2] / img.shape[1]
        bb[1::2] = bb[1::2] / img.shape[0]
        bb = marginise_to_norm_range(bb)

        img = cv2.resize(img, (img.shape[1] * 300 // img.shape[0], 300))

        image = ".".join([image.split(".")[0], "png"])
        img_path = pj(out_path, image)
        cv2.imwrite(img_path, img)

        filenames.append(image)
        bbs.append(bb)
    df_out = pd.DataFrame(dict(filename=filenames, bb=bbs))
    df_out.to_csv(out_csv)


if __name__ == '__main__':
    normalise_images()
    tf.app.run()
