"""
"""
import torch

from cow_tus.models.modules.i3d import I3D


class I3DEncoder(I3D):
    """
    """
    def __init__(self,
                 modality='gray',
                 attention=False,
                 dropout_prob=0,
                 weights_path=None):
        """ I3D encoder that encodes a stack of images into a block of size (length x  7 x 7)
        """

        # Three possible modalities: rgb, flow, gray
        assert modality in {'rgb', 'flow', 'gray'}, 'Invalid modality.'
        super().__init__(modality, dropout_prob, weights_path=weights_path)

        # remove unused layers
        self.conv3d_0c_1x1 = None
        self.avg_pool = None

    def forward(self, input_tensor):
        """
        Encodes a stack of images.
        args:
            input_tensor    (torch.tensor) shape (batch_size, length, height=200, width=200, channels)
        return:
            output (torch.tensor)   shape (batch_size, 1024, ~length/8, 7, 7)
        """
        input_tensor = input_tensor.permute((0, 4, 1, 2, 3))
        out = self.conv3d_1a_7x7(input_tensor)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        return out