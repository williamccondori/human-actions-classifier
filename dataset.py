import os
import cv2
import darknet

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

def convert(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def detect_people(image):
    inputi = image

    config_file = 'cfg/yolov4.cfg'
    weights_file = 'yolov4.weights'
    data_file = 'data/coco.data'

    height, width, _ = image.shape
    
    network = darknet.load_net_custom(config_file.encode('ascii'), weights_file.encode('ascii'), 0, 1)

    meta = darknet.load_meta(data_file.encode('ascii'))
    darknet_image = darknet.make_image(darknet.network_width(network), darknet.network_height(network),3)
    image = cv2.resize(image, (darknet.network_width(network), darknet.network_height(network)), interpolation=cv2.INTER_LINEAR)
    wi = darknet.network_width(network)
    he = darknet.network_height(network)
    darknet.copy_image_from_bytes(darknet_image, image.tobytes())

    results = darknet.detect_image(network, meta, darknet_image, thresh=0.25)

    for result in results:
        item, confidence, bounding_box = result
        if item == 'person':
            result = convert(bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3])
            # xc + rx(xo -xc)
            rx = width / wi
            ry = height / he
            xmin = result[0] * rx
            ymin = result[1] * ry
            xmax = result[2] * rx
            ymax = result[3] * ry
            cv2.rectangle(inputi,(int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), 1)
            cv2.putText(inputi, str(item.decode('utf-8')), (int(xmin), int(ymin) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (0,255,0), 1)

    result = cv2.cvtColor(inputi, cv2.COLOR_BGR2RGB)
    cv2.imwrite('resultado.jpg', result)

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


if __name__ == "__main__":
    # image = Image.open('test/S020_F_COUG_WLK_RGT_HF_17.jpg')
    #image = Image.open('test/person.jpg')
    image = cv2.cvtColor(cv2.imread('test/person.jpg'), cv2.COLOR_BGR2RGB)
    detect_people(image)
