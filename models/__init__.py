
from .pinpoint3d import build_PinPoint3D

from .criterion import build_mask_criterion

def build_model(args):
    if hasattr(args, 'model_type') and args.model_type == 'pinpoint3D':
        return build_PinPoint3D(args)

def build_criterion(args):
    if hasattr(args, 'model_type') and args.model_type == 'pinpoint3D':
        return build_mask_criterion(args)
    
