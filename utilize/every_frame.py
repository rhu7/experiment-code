import os

import cv2
import logging

# log information settings
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def extract_frame(video_path, save_path):
    vc = cv2.VideoCapture(video_path)  # import video files

    if vc.isOpened():  # determine whether to open normally
        ret = True
    else:
        ret = False

    count = 1  # count the number of pictures

    while ret:  # loop read video frame
        ret, frame = vc.read()
        frame_path = os.path.join(save_path, 'img_{:05d}.jpg'.format(count))
        if ret:
            cv2.imwrite(frame_path, frame)  # store operation
            count += 1
        # cv2.waitKey(1)
    logging.info("Total_frame_num：" + str(count-1))
    vc.release()

if __name__ == "__main__":
    src = r'/data2/wangsw/dataset/WJJ_label-1.0'
    dst = r'/data2/wangsw/dataset/WJJ_label_rgb'
    catogs = os.listdir(src)
    for catog in catogs:
        src_path = os.path.join(src, catog)
        dst_path = os.path.join(dst, catog)

        video_names = os.listdir(src_path)
        for name in video_names:
            video_path = os.path.join(src_path, name)
            save_path = os.path.join(dst_path, os.path.splitext(name)[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            extract_frame(video_path, save_path)