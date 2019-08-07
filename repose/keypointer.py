import numpy as np

import sys
import os

from edflow.custom_logging import get_logger


PYTHON_OPENPOSE = os.environ.get("PYTHON_OPENPOSE",
                                 "/export/home/pesser/src/openpose_cpu/build_opencv/"
                                 "python/openpose/")
MODEL_OPENPOSE = os.environ.get("MODEL_OPENPOSE",
                                "/export/home/pesser/src/openpose_cpu/models/")


logger = get_logger(__name__)


class KeyPointer(object):
    joint_order = [
            'cnose', 'cneck',
            'rshoulder', 'relbow', 'rwrist',
            'lshoulder', 'lelbow', 'lwrist',
            'chip',
            'rhip', 'rknee', 'rankle',
            'lhip', 'lknee', 'lankle',
            'reye',
            'leye',
            'rear',
            'lear',
            'lbigtoe', 'lsmalltoe', 'lheel',
            'rbigtoe', 'rsmalltoe', 'rheel',  # TODO I guess it stops here?
            'rankle', 'rknee', 'rhip', 'lhip', 'lknee', 'lankle',
            'rwrist', 'relbow', 'rshoulder', 'lshoulder', 'lelbow',
            'lwrist', 'cnose', 'leye', 'reye']

    def __init__(self, gpu_id=0, n_people=1, net_resolution=(256, 256),
                 scale_number=4, scale_gap=0.25, keypoint_scale=3):
        '''Sets up an openpose object, which can be shared.

        Args:
            gpu_id (int): Id of the gpu to be used. Default to `0`. No
                effect when used with CPU_ONLY build of OpenPose.
            n_people (int): Maximum number of people to be detected.
                Ignored.
            net_resolution (2-tuple of ints): Resolution (width, height) at
                which model runs. Higher resolution might give better results
                but probably only for images containing small people. Should
                match AR of input image,
            scale_number (int): Number of scales to average over.
            scale_gap (float): Ratio between different scales.
            keypoint_scale (int): Scaling of keypoints. From OpenPose:
                0 to scale it to the original source resolution;
                1 to scale it to the net output size (set with net_resolution);
                2 to scale it to the final output size (set with resolution);
                3 to scale it in the range [0,1], where (0,0) would be the
                top-left corner of the image, and (1,1) the bottom-right
                one; and
                4 for range [-1,1], where (-1,-1) would be the
                top-left corner of the image, and (1,1) the bottom-right
                one.

        Returns:
            object: The Pose estimator
        '''

        sys.path.append(PYTHON_OPENPOSE)
        from openpose import OpenPose
        gpu_id = int(os.environ.get("GPUID_OPENPOSE", gpu_id))

        params = {'logging_level': 3,
                  'net_resolution': "x".join(map(str, net_resolution)),
                  'model_pose': 'BODY_25',
                  'scale_number': scale_number,
                  'scale_gap': scale_gap,
                  'num_gpu_start': gpu_id,
                  'default_model_folder': MODEL_OPENPOSE,
                  # 'number_people_max': n_people, # seems to be ignored
                  # rendering related - not so interesting for us  but
                  # required
                  'output_resolution': '-1x-1',
                  'alpha_pose': 0.6,
                  'render_threshold': 0.05,
                  'disable_blending': False,
                  }
        self.net_resolution = net_resolution
        self.keypoint_scale = keypoint_scale
        self.n_keypoints = 25

        self.openpose = OpenPose(params)

    def rescale_keypoints(self, image, keypoints):
        """OpenPose python wrapper does not support keypoint_scale so we do
        it ourselves"""
        # reshape if no detections
        if keypoints.size == 0 and keypoints.shape[1:] != (
                self.n_keypoints, 3):
            keypoints = np.reshape(keypoints, (0, self.n_keypoints, 3))
        if self.keypoint_scale == 0:
            return keypoints

        # rescale to [0,1]
        h, w = image.shape[:2]
        keypoints = keypoints / np.array([[[(w-1), (h-1), 1.0]]])

        if self.keypoint_scale == 1:
            net_w, net_h = self.net_resolution
            return keypoints * np.array([[[(net_w-1), (net_h-1), 1.0]]])
        elif self.keypoint_scale == 2:
            raise NotImplementedError("Python wrapper does not support "
                                      "resolution.")
        elif self.keypoint_scale == 3:
            return keypoints
        elif self.keypoint_scale == 4:
            return np.array([[[2.0, 2.0, 1.0]]]
                            )*keypoints - np.array([[[1.0, 1.0, 0.0]]])
        raise NotImplementedError(
            "Invalid keypoint_scale: {}".format(self.keypoint_scale)
        )

    def __call__(self, image, render=False):
        '''Returns the keypoints in an image for any number of people in it.

        Args:
            image (np.array): 3d Array. (height, width, 3) - last axis is
                assumed to be BGR as this is based on Caffe!
            render (bool): Whether or not to return rendered image as well.

        Returns:
            np.array: keypoints as specified in the openpose documentation.
                Array of shape [n_persons, n_joints, 3] where last axis is
                x,y,confidence.
        '''

        if not image.flags['C_CONTIGUOUS']:
            raise ValueError("OpenPose can not handle views!")
        result = self.openpose.forward(image, render)
        if render:
            keypoints, rendered_image = result
        else:
            keypoints = result
        keypoints = self.rescale_keypoints(image, keypoints)
        if render:
            return keypoints, rendered_image
        else:
            return keypoints


if __name__ == '__main__':
    K = KeyPointer()
    K(np.ones([100, 100, 3], dtype=np.uint8))
