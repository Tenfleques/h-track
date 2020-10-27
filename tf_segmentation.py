import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import time
import cv2

from tools import tools

# tf.enable_eager_execution()

class BodyPixSegmentation:

    def __init__(self, model=BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16) -> None:
        self.bodypix_model = load_model(download_model(
            model
        ))

    def get_bbox(self, frame, mask_threshold=.75, interactive=False):
        image_array = tf.keras.preprocessing.image.img_to_array(frame)
        # print(image_array.shape)

        # start_time = time.time()
        result = self.bodypix_model.predict_single(image_array)
        # print("[INFO] predict single time is {:0.2f} ".format(time.time() - start_time))

        mask = result.get_mask(threshold=mask_threshold).numpy()

        print(mask.shape)

        # start_time = time.time()
        (x1, y1), (x2, y2) = tools.fast_get_bbox(mask)

        if interactive:
            color = (255, 0, 0)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.imshow("frame", frame)
            cv2.waitKey(5000)

        # tf.keras.preprocessing.image.save_img(
        #     './output/test-mask.jpg',
        #     mask.numpy()
        # )

        return (x1, y1), (x2, y2)

def process_video():
    pass

if __name__ == "__main__":
    image_path = './input/test.jpg'
    frame = cv2.imread(image_path)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    body_pix_seg = BodyPixSegmentation()

    body_pix_seg.get_bbox(frame)
