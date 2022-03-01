import cv2
import numpy as np

from pyturbo import Stage


class SIFTFeature(Stage):

    """
    Input: frame [H x W x C]
    Output: SIFT feature [N x D]
    """

    def allocate_resource(self, resources, *, num_features=32):
        self.num_features = num_features
        self.sift = None
        return [resources]

    def reset(self):
        if self.sift is None:
            self.sift = cv2.SIFT_create(self.num_features)

    def extract_sift_feature(self, frame: np.ndarray) -> np.ndarray:
        """
        frame: [H x W x C]

        Return: Feature for N key points, [N x 128]
        """
        # Extract SIFT feature for the current frame
        # Use self.sift.detectAndCompute
        # Remember to handle when it returns None
        self.reset()
        keypoints, descriptor = self.sift.detectAndCompute(frame, None)
        if descriptor is None:
            descriptor = np.zeros((self.num_features, 128))
        return descriptor

    def process(self, task):
        task.start(self)
        frame = task.content
        feature = self.extract_sift_feature(frame)
        assert feature is not None and isinstance(feature, np.ndarray)
        assert feature.shape[1] == 128
        return task.finish(feature)
