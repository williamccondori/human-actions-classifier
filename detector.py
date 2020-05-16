import os
import cv2
import darknet

def detector(image, image_size, darknet_model, darknet_meta, darknet_image, darknet_size, log=False):

    # Load darknet image.
    image_resized = cv2.resize(
        image, darknet_size, interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

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
            bounding_box = rescale(image_size, darknet_size, bounding_box)
            start_point = (bounding_box[0], bounding_box[1])
            end_point = (bounding_box[2], bounding_box[3])

            # Add indicators.
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 1)
            cv2.putText(image, f'{class_id}: {confidence}', (
                bounding_box[0], bounding_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .2, (0, 255, 0), 1)

    result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return result


if __name__ == "__main__":
    detector()
