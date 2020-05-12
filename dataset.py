import os
import cv2


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
    main()
