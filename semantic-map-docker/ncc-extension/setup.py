
import setuptools
from setuptools import setup
import torch.utils
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='ncc',
      ext_modules=[CppExtension('ncc', ['ncc.cpp'])],
      cmdclass={'build_ext': BuildExtension})

setuptools.Extension(
   name='ncc',
   sources=['ncc.cpp'],
   include_dirs=torch.utils.cpp_extension.include_paths(),
   language='c++')

