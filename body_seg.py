import os
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
from utils import load_graph_model, get_input_tensors, get_output_tensors
import tensorflow as tf
import cv2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
KEYPOINT_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

KEYPOINT_IDS = {name: id for id, name in enumerate(KEYPOINT_NAMES)}
CONNECTED_KEYPOINTS_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]

CONNECTED_KEYPOINT_INDICES = [(KEYPOINT_IDS[a], KEYPOINT_IDS[b])
                              for a, b in CONNECTED_KEYPOINTS_NAMES]


PART_CHANNELS = [
    'left_face',
    'right_face',
    'left_upper_arm_front',
    'left_upper_arm_back',
    'right_upper_arm_front',
    'right_upper_arm_back',
    'left_lower_arm_front',
    'left_lower_arm_back',
    'right_lower_arm_front',
    'right_lower_arm_back',
    'left_hand',
    'right_hand',
    'torso_front',
    'torso_back',
    'left_upper_leg_front',
    'left_upper_leg_back',
    'right_upper_leg_front',
    'right_upper_leg_back',
    'left_lower_leg_front',
    'left_lower_leg_back',
    'right_lower_leg_front',
    'right_lower_leg_back',
    'left_feet',
    'right_feet'
]


class BodySeg:
    # CONSTANTS
    output_stride = 16
    segments = None
    results = None
    mask = None
    segmentation_mask = None
    part_heatmaps = None
    key_point_positions = []
# source img is of size int(imgSide) // self.output_stride) * self.output_stride + 1 where side = [width, height]

    def __init__(self, img, model_path='./bodypix_resnet50_float_model-stride16/model.json', output_stride=16):
        print("[INFO] Loading model...")
        self.graph = load_graph_model(model_path)
        print("[INFO] Loaded model...")
        self.output_stride = output_stride

        self.img = img

        # Get input and output tensors
        self.input_tensor_names = get_input_tensors(self.graph)
        print(self.input_tensor_names)
        self.output_tensor_names = get_output_tensors(self.graph)
        print(self.output_tensor_names)
        self.input_tensor = self.graph.get_tensor_by_name(self.input_tensor_names[0])

    @staticmethod
    def get_bounding_box(key_point_positions, offset=(10, 10, 10, 10)):
        min_x = math.inf
        min_y = math.inf
        max_x = - math.inf
        max_y = -math.inf
        for x, y in key_point_positions:
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
        return (min_x - offset[0], min_y - offset[1]), (max_x + offset[2], max_y + offset[3])

    def process_image(self):
        # Preprocessing Image
        # For Res-net
        if any('resnet_v1' in name for name in self.output_tensor_names):
            # add image-net mean - extracted from body-pix source
            m = np.array([-123.15, -115.90, -103.06])
            self.img = np.add(self.img, m)
        # For Mobile-net
        elif any('MobilenetV1' in name for name in self.output_tensor_names):
            self.img = (self.img / 127.5) - 1
        else:
            print('Unknown Model')
        print("done.\nRunning inference...", end="")

        sample_image = self.img[tf.newaxis, ...]

        # evaluate the loaded model directly
        with tf.compat.v1.Session(graph=self.graph) as sess:
            self.results = sess.run(self.output_tensor_names, feed_dict={
                self.input_tensor: sample_image})

        # assert len(results) == 8
        print("done. {} outputs received".format(len(self.results)))  # should be 8 outputs

        self.set_float_segments(False)

        for idx, name in enumerate(self.output_tensor_names):
            if 'displacement_bwd' in name:
                self.displacement_bwd()
            elif 'float_heatmaps' in name:
                self.heatmaps = np.squeeze(self.results[idx], 0)
                print('heatmaps', name, self.heatmaps.shape)
            elif 'float_long_offsets' in name:
                self.long_offsets = np.squeeze(self.results[idx], 0)
                print('long_offsets', name, self.long_offsets.shape)
            elif 'float_short_offsets' in name:
                self.offsets = np.squeeze(self.results[idx], 0)
                print('offsets', name, self.offsets.shape)
            elif 'float_part_heatmaps' in name:
                self.part_heatmaps = np.squeeze(self.results[idx], 0)
                print('part_heatmaps', name,  self.part_heatmaps.shape)
            elif 'float_part_offsets' in name:
                self.part_offsets = np.squeeze(self.results[idx], 0)
                print('partOffsets', name, self.part_offsets.shape)

    def __get_results_by_key(self, key):
        # checks is key is found in the results return
        if self.results is None:
            return None

        k = -1
        for i in self.output_tensor_names:
            k += 1
            if key in i:
                break

        if k == -1:
            print("[ERROR] {} not found".format(key))
            return None

        return self.results[k]

    def set_float_segments(self, plot_me=False):

        output_tensor = self.__get_results_by_key("float_segments")
        if output_tensor is None:
            return

        self.segments = np.squeeze(output_tensor, 0)
        # Segmentation MASk
        segmentation_threshold = 0.7
        segment_scores = tf.sigmoid(self.segments)
        self.mask = tf.math.greater(segment_scores, tf.constant(segmentation_threshold))
        segmentation_mask = tf.dtypes.cast(self.mask, tf.int32)

        self.segmentation_mask = np.reshape(
            segmentation_mask, (segmentation_mask.shape[0], segmentation_mask.shape[1]))

        if plot_me:
            self.plot("seg_mask")

    def plot(self, mask_name="seg_mask"):
        keys = {
            "seg_mask": {
                "title": 'Segmentation Mask',
                "value": self.segmentation_mask * self.output_stride
            }
        }
        if mask_name not in keys.keys():
            return None

        plt.clf()
        plt.title(keys[mask_name].get("title"))
        plt.ylabel('y')
        plt.xlabel('x')
        plt.imshow(keys[mask_name].get("value"))
        plt.show()

    def displacement_bwd(self):
        output_tensor = self.__get_results_by_key("displacement_bwd")
        if output_tensor is None:
            return
        print('displacement_bwd', "displacement_bwd", output_tensor.shape)

    def process_part_heatmaps(self):
        print(self.output_tensor_names)

        for idx, name in enumerate(self.output_tensor_names):
            if 'float_part_heatmaps' in name:
                self.part_heatmaps = np.squeeze(self.results[idx], 0)
                print('part_heatmaps', name, self.part_heatmaps.shape)
                break

        # Part Heatmaps, PartOffsets,
        if self.mask is not None:
            for i in range(self.part_heatmaps.shape[2]):

                heatmap = self.part_heatmaps[:, :, i]  # First Heat map
                # heatmap[np.logical_not(tf.math.reduce_any(self.mask, axis=-1).numpy())] = -1
                # Set portions of heatmap where person is not present in segmentation mask, set value to -1
                print('Heatmap: ' + PART_CHANNELS[i], heatmap.shape)
                # cv2.imshow('Heatmap: ' + PART_CHANNELS[i], heatmap)
                # cv2.waitKey(1000)

                # SHOW HEATMAPS
                # plt.clf()
                # plt.title('Heatmap: ' + PART_CHANNELS[i])
                # plt.ylabel('y')
                # plt.xlabel('x')
                # plt.imshow(heatmap * self.output_stride)
                # plt.show()
                #
                # heatmap_sigmoid = tf.sigmoid(heatmap)
                # y_heat, x_heat = np.unravel_index(
                #     np.argmax(heatmap_sigmoid, axis=None), heatmap_sigmoid.shape)

                # Offset Corresponding to heatmap x and y
                # x_offset = self.part_offsets[y_heat, x_heat, i]
                # y_offset = self.part_offsets[y_heat, x_heat, self.part_heatmaps.shape[2]+i]
                #
                # key_x = x_heat * self.output_stride + x_offset
                # key_y = y_heat * self.output_stride + y_offset

    def evaluate_outputs(self):
        # Draw Segmented Output
        mask_img = Image.fromarray(self.segmentation_mask * 255)
        mask_img = mask_img.resize(
            (self.img.shape[1], self.img.shape[0]), Image.LANCZOS).convert("RGB")
        mask_img = tf.keras.preprocessing.image.img_to_array(
            mask_img, dtype=np.uint8)

        segmentation_mask_inv = np.bitwise_not(mask_img)
        fg = np.bitwise_and(self.img.astype(np.uint8), np.array(
            mask_img))
        plt.title('Foreground Segmentation')
        plt.imshow(fg)
        plt.show()
        bg = np.bitwise_and(self.img.astype(np.uint8), np.array(
            segmentation_mask_inv))
        plt.title('Background Segmentation')
        plt.imshow(bg)
        plt.show()

        self.process_part_heatmaps()
        #
        # POSE ESTIMATION
        key_scores = []
        for i in range(self.heatmaps.shape[2]):
            heatmap = self.heatmaps[:, :, i]  # First Heat map
            # SHOW HEATMAPS

            # plt.clf()
            # plt.title('Heatmap' + str(i) + KEYPOINT_NAMES[i])
            # plt.ylabel('y')
            # plt.xlabel('x')
            # plt.imshow(heatmap * self.output_stride)
            # plt.show()

            heatmap_sigmoid = tf.sigmoid(heatmap)
            y_heat, x_heat = np.unravel_index(
                np.argmax(heatmap_sigmoid, axis=None), heatmap_sigmoid.shape)

            key_scores.append(heatmap_sigmoid[y_heat, x_heat].numpy())
            # Offset Corresponding to heatmap x and y
            x_offset = self.offsets[y_heat, x_heat, i]
            y_offset = self.offsets[y_heat, x_heat, self.heatmaps.shape[2]+i]

            key_x = x_heat * self.output_stride + x_offset
            key_y = y_heat * self.output_stride + y_offset
            self.key_point_positions.append([key_x, key_y])

        print('keypointPositions', np.asarray(self.key_point_positions).shape)
        print('keyScores', np.asarray(key_scores).shape)

        # PRINT KEYPOINT CONFIDENCE SCORES
        print("Keypoint Confidence Score")
        for i, score in enumerate(key_scores):
            print(KEYPOINT_NAMES[i], score)

        # PRINT POSE CONFIDENCE SCORE
        print("Pose Confidence Score", np.mean(np.asarray(key_scores)))

        # # Get Bounding BOX
        (xmin, ymin), (xmax, ymax) = self.get_bounding_box(
            self.key_point_positions, offset=(0, 0, 0, 0))
        print("Bounding Box xmin, ymin, xmax, ymax format: ", xmin, ymin, xmax, ymax)

        # Show Bounding BOX
        # # Show all keypoints
        #
        cv2.imshow("image", self.img)
        cv2.waitKey(5000)

        # x_points = []
        # y_points = []
        # for i, [x, y] in enumerate(self.key_point_positions):
        #     x_points.append(x)
        #     y_points.append(y)
        # plt.scatter(x=x_points, y=y_points, c='r', s=40)
        # plt.show()
        # #
        # #
        # # # DEBUG KEYPOINTS
        # # #  Show Each Keypoint and it's name
        # # '''
        # for i, [x, y] in enumerate(self.key_point_positions):
        #     plt.figure(i)
        #     plt.title('keypoint' + str(i) + KEYPOINT_NAMES[i])
        #     implot = plt.imshow(self.img)
        #
        #     plt.scatter(x=[x], y=[y], c='r', s=40)
        #     plt.show()
        # # '''
        # #
        # # # SHOW CONNECTED KEYPOINTS
        # plt.figure(20)
        # for pt1, pt2 in CONNECTED_KEYPOINT_INDICES:
        #     plt.title('connection points')
        #     implot = plt.imshow(self.img)
        #     plt.plot((self.key_point_positions[pt1][0], self.key_point_positions[pt2][0]), (
        #         self.key_point_positions[pt1][1], self.key_point_positions[pt2][1]), 'ro-', linewidth=2, markersize=5)
        # plt.show()
