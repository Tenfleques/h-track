import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import time

bodypix_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
))

start_time = time.time()
image = tf.keras.preprocessing.image.load_img(
    './images/test.jpg'
)
print("[INFO] preprocessing time is {:0.2f} ".format(time.time() - start_time))

image_array = tf.keras.preprocessing.image.img_to_array(image)
start_time = time.time()
result = bodypix_model.predict_single(image_array)
print("[INFO] predict single time is {:0.2f} ".format(time.time() - start_time))

mask = result.get_mask(threshold=0.75)

tf.keras.preprocessing.image.save_img(
    './images/test-mask.jpg',
    mask
)

# colored_mask = result.get_colored_mask(mask)
# tf.keras.preprocessing.image.save_img(
#     './images/test-colored-mask.jpg',
#     colored_mask
# )
