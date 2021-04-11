from setuptools import setup

setup(name='example',
      version='1.0',
      description='An example package for Harvard PHYS201',
      url='http://github.com/phys201/example',
      author='ralex0',
      author_email='ralex0@users.noreply.github.com',
      license='GPLv3',
<<<<<<< Updated upstream
      packages=['example'],
      install_requires=['numpy'])
=======
      packages=['qd_laser_dynamics'],
      install_requires=[
          'numpy'
          'pandas',
          'scipy',
          'matplotlib',
          'emcee',
          'nose',
          'unittest'
          ]
      test_suite = 'nose.collector',
      tests_require = ['nose'],
      )
>>>>>>> Stashed changes
