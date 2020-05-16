import os
import cv2
import darknet


def test_image():

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


    # Init YOLO model (COCO).
    darknet_meta = darknet.load_meta('data/coco.data'.encode('ascii'))
    darknet_model = darknet.load_net_custom(
        'cfg/yolov4.cfg'.encode('ascii'), 'yolov4.weights'.encode('ascii'), 0, 1)

    darknet_size = (darknet.network_width(darknet_model),
                    darknet.network_height(darknet_model))
    darknet_image = darknet.make_image(darknet_size[0], darknet_size[1], 3)


if __name__ == "__main__":
    main()
