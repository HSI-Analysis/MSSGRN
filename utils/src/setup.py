from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

ext_modules = []


# Check if CUDA is available
if os.environ.get('CUDA_HOME') is not None:
    ext_modules.append(
        CUDAExtension('pair_wise_distance_cuda', [
            'pair_wise_distance_cuda_source.cu',
        ])
    )

# setup(
#     name='pair_wise_distance',
#     ext_modules=[
#         CUDAExtension('pair_wise_distance_cuda', [
#             'pair_wise_distance_cuda_source.cu',
#         ])
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })


setup(
    name='pair_wise_distance',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)