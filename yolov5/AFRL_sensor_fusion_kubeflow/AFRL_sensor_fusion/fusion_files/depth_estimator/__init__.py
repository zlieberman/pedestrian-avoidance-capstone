from .monodepth2 import Monodepth2

__all__ = ['build_Monodepth2']

def build_Monodepth2(model_name, use_cuda):
    return Monodepth2(model_name, use_cuda)
