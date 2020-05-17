import cv2
import darknet
import detector as yolo


def main():

    image = cv2.imread('test/test2.jpg', cv2.COLOR_BGR2RGB)

    width = image.size().width
    height = image.size().height

    darknet_meta = darknet.load_meta('data/coco.data'.encode('ascii'))
    darknet_model = darknet.load_net_custom(
        'cfg/yolov4.cfg'.encode('ascii'), 'yolov4.weights'.encode('ascii'), 0, 1)

    darknet_size = (darknet.network_width(darknet_model),
                    darknet.network_height(darknet_model))
    darknet_image = darknet.make_image(darknet_size[0], darknet_size[1], 3)

    image_result = yolo.detector(
        image, (width, height), darknet_model, darknet_meta, darknet_image, darknet_size,  False)


if __name__ == "__main__":
    main()
