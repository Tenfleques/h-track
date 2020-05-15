from body_seg import BodySeg
import numpy as np
import cv2

if __name__ == "__main__":
    image = cv2.imread("./images/test.jpg")
    win_name = "preview"
    # let's take a look at the image that loaded
    # cv2.imshow(win_name, image)
    # # keep showing image till use hit q to exit
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     cv2.destroyWindow(win_name)
    # good image preview shows dear image, now time to investigate segmentation

    # setting image to RGB
    image = image[..., ::-1]

    # we like the other defaults
    body_seg = BodySeg(image)
    body_seg.process_image()
