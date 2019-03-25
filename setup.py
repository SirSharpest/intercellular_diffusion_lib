from setuptools import setup

package = 'intercellular_diffusion_lib'
version = '0.1'

setup(name=package,
      version=version,
      description="A library for analysing intercellular diffusion",
      url='https://www.github.com/sirsharpest/intercellular_diffusion_lib',
      author='Nathan Hughes',
      packages=['intercellular_diffusion_lib'],
      install_requires=['numpy', 'pandas'],
      zip_safe=True)
