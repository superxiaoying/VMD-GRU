import torch
import torch.nn as nn

class ExternalMasking(nn.Module):
    def __init__(self, mask_value=0.):
        super(ExternalMasking, self).__init__()
        self.mask_value = mask_value

    def forward(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('Inputs to ExternalMasking should be a list of 2 tensors.')
        boolean_mask = (inputs[-1] != self.mask_value).any(dim=-1, keepdim=True)
        return inputs[0] * boolean_mask.float()

    def extra_repr(self) -> str:
        return 'mask_value={}'.format(self.mask_value)