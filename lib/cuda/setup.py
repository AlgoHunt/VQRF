from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules=[
    CUDAExtension(
            name='dvgo_cu.adam_upd_cuda',
            sources=['adam_upd_kernel.cu',
                     'adam_upd.cpp'],
                   
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']}),
    CUDAExtension(
            name='dvgo_cu.render_utils_cuda',
            sources=['render_utils_kernel.cu',
                    'render_utils.cpp'],
                   
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']}),
    CUDAExtension(
            name='dvgo_cu.total_variation_cuda',
            sources=['total_variation_kernel.cu',
                    'total_variation.cpp'],
                   
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']}),
     CUDAExtension(
            name='dvgo_cu.ub360_utils',
            sources=['ub360_utils_kernel.cu',
                    'ub360_utils.cpp'],
                   
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']})

]

setup(
        name='dvgo_cu',
        ext_modules = ext_modules,
        cmdclass={
            'build_ext': BuildExtension
        })

 # 'cuda/render_utils_kernel.cu' ,
                    # 'cuda/render_utils.cpp',
                    # 'cuda/total_variation_kernel.cu' ,
                    # 'cuda/total_variation.cpp'],