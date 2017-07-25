from setuptools import setup

setup(name='pyrds',
      version='1.0',
      description='python implementation of a finitie volume discretization to solve the reaction diffusion equation',
      author='Matthias Zimmermann',
      author_email='matthias.zimmermann@eawag.ch',
      packages=['pyrds'],
      install_requires=['numpy>=1.13', 'scipy>=0.19'],
      zip_safe=False)
