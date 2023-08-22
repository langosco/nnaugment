from setuptools import setup

setup(name='nnaugment',  # for pip show, pip uninstall, etc
      version='0.0.1',
      author="Lauro Langosco",
      description="Data augmentation for datasets of neural networks via parameter permutations.",

      packages=["nnaugment"], # for `import nnaugment`
      install_requires=[],
      )
