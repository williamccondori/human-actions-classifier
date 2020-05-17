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
