import numpy as np
import torch
from pyturbo import Stage
from torch.backends import cudnn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class CNNFeature(Stage):

    """
    Input: frame [H x W x C]
    Output: CNN feature [N x D]
    """

    def allocate_resource(self, resources, *, model_name='resnet18',
                          node_name='avgpool', replica_per_gpu=1):
        self.model_name = model_name
        self.node_name = node_name
        gpus = resources.get('gpu')
        self.num_gpus = len(gpus)
        if len(gpus) > 0:
            return resources.split(len(gpus)) * replica_per_gpu
        return [resources]

    def reset(self):
        gpu_ids = self.current_resource.get('gpu', 1)
        if len(gpu_ids) >= 1:
            self.device = 'cuda:%d' % (gpu_ids[0])
            cudnn.fastest = True
            cudnn.benchmark = True
        else:
            self.device = 'cpu'
            self.logger.warn('No available GPUs, running on CPU.')
        base_model = getattr(models, self.model_name)(pretrained=True)
        self.model = create_feature_extractor(
            base_model, {self.node_name: 'feature'})
        self.model = self.model.to(self.device).eval()

    def extract_cnn_feature(self, frame: np.ndarray) -> np.ndarray:
        """
        frame: [H x W x C] in uint8 [0, 255]

        Return: Feature, [N x D]
        """
        # TODO: extract CNN feature for the frame
        # Use self.model, whose input is [B x C x H x W] in float [0, 1]
        # Recommended to use with torch.no_grad()
        raise NotImplementedError
    
    def process(self, task):
        task.start(self)
        frame = task.content
        feature = self.extract_cnn_feature(frame)
        assert feature is not None and isinstance(feature, np.ndarray)
        return task.finish(feature)
