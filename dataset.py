import os
import cv2
import darknet

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


def convert(bounding_box):
    x_min = int(round(bounding_box[0] - (bounding_box[2] / 2)))
    x_max = int(round(bounding_box[0] + (bounding_box[2] / 2)))
    y_min = int(round(bounding_box[1] - (bounding_box[3] / 2)))
    y_max = int(round(bounding_box[1] + (bounding_box[3] / 2)))
    return (x_min, y_min, x_max, y_max)


def rescale(frame_size, draknet_size, bounding_box):
    factor_x = frame_size[0] / draknet_size[0]
    factor_y = frame_size[1] / draknet_size[1]
    x_min = int(bounding_box[0] * factor_x)
    y_min = int(bounding_box[1] * factor_y)
    x_max = int(bounding_box[2] * factor_x)
    y_max = int(bounding_box[3] * factor_y)
    return (x_min, y_min, x_max, y_max)


def detect_people(image):
    inputi = image

    config_file = 'cfg/yolov4.cfg'
    weights_file = 'yolov4.weights'
    data_file = 'data/coco.data'

    height, width, _ = image.shape

    network = darknet.load_net_custom(config_file.encode(
        'ascii'), weights_file.encode('ascii'), 0, 1)

    meta = darknet.load_meta(data_file.encode('ascii'))
    darknet_image = darknet.make_image(darknet.network_width(
        network), darknet.network_height(network), 3)
    image = cv2.resize(image, (darknet.network_width(
        network), darknet.network_height(network)), interpolation=cv2.INTER_LINEAR)
    wi = darknet.network_width(network)
    he = darknet.network_height(network)
    darknet.copy_image_from_bytes(darknet_image, image.tobytes())

    results = darknet.detect_image(network, meta, darknet_image, thresh=0.25)

    for result in results:
        item, confidence, bounding_box = result
        item = item.decode('utf-8')
        if item == 'person':
            print(f'{item} {confidence}')
            result = convert(
                bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3])
            # xc + rx(xo -xc)
            rx = width / wi
            ry = height / he
            xmin = result[0] * rx
            ymin = result[1] * ry
            xmax = result[2] * rx
            ymax = result[3] * ry
            cv2.rectangle(inputi, (int(xmin), int(ymin)),
                          (int(xmax), int(ymax)), (0, 255, 0), 1)
            cv2.putText(inputi, item, (int(xmin), int(ymin) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)

    result = cv2.cvtColor(inputi, cv2.COLOR_BGR2RGB)
    return result


def detect_video(frame, frame_size, darknet_model, darknet_meta, darknet_image, darknet_size, log=False):

    # Load darknet image.
    frame_resized = cv2.resize(
        frame, darknet_size, interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

    # Detect.
    results = darknet.detect_image(
        darknet_model, darknet_meta, darknet_image, thresh=0.25)

    for result in results:
        class_id, confidence, bounding_box = result
        class_id = class_id.decode('utf-8')
        if class_id == 'person':
            if log:
                print(f'{class_id}: {confidence}')
            # Convert from YOLO format.
            bounding_box = convert(bounding_box)

            # Rescaling the bounding boxes.
            bounding_box = rescale(frame_size, darknet_size, bounding_box)
            start_point = (bounding_box[0], bounding_box[1])
            end_point = (bounding_box[2], bounding_box[3])

            # Add indicators.
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 1)
            cv2.putText(frame, f'{class_id}: {confidence}', (
                bounding_box[0], bounding_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .2, (0, 255, 0), 1)

    result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return result


def main():
    video_file_path = 'C:/Users/William/Downloads/biisc/biisc/videos'
    for video_file_name in os.listdir(video_file_path):
        name = video_file_name.split('.')[0]
        name_components = name.split('_')
        video = cv2.VideoCapture(f'{video_file_path}/{video_file_name}')
        success, image = video.read()
        count = 0
        while success:
            if not os.path.exists(f'images/{name_components[2]}'):
                print(f'Creating folder: {name_components[2]}')
                os.makedirs(f'images/{name_components[2]}')
            cv2.imwrite(
                f'images/{name_components[2]}/{name}_{count}.jpg', image)
            success, image = video.read()
            count += 1


def test_video(input_file, output_file):
    # Load video.
    capture = cv2.VideoCapture(input_file)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video frames={length} width={width} height={height}')

    # Init YOLO model (COCO).
    darknet_meta = darknet.load_meta('data/coco.data'.encode('ascii'))
    darknet_model = darknet.load_net_custom(
        'cfg/yolov4.cfg'.encode('ascii'), 'yolov4.weights'.encode('ascii'), 0, 1)

    darknet_size = (darknet.network_width(darknet_model),
                    darknet.network_height(darknet_model))
    darknet_image = darknet.make_image(darknet_size[0], darknet_size[1], 3)

    # Generating video output.
    codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    framerate = 30
    resolution = (width, height)
    output_video = cv2.VideoWriter(output_file, codec, framerate, resolution)

    if capture.isOpened():
        ret, frame = capture.read()

    console = '.'
    while ret:
        ret, frame = capture.read()
        result_frame = detect_video(
            frame, (width, height), darknet_model, darknet_meta, darknet_image, darknet_size)
        output_video.write(frame)
        print(console + console)

    capture.release()
    output_video.release()


if __name__ == "__main__":
    # image = Image.open('test/S020_F_COUG_WLK_RGT_HF_17.jpg')
    #image = Image.open('test/person.jpg')
    # image = cv2.cvtColor(cv2.imread('test/test1.jpg'), cv2.COLOR_BGR2RGB)
    # result = detect_people(image)
    # cv2.imwrite('result.jpg', result)

    test_video('test/test4.mp4', 'result.mp4')
