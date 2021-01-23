from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension("cython_nms", ["cython_nms.pyx"])]
setup(
  name = 'nms',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)