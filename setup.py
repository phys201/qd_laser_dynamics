from setuptools import setup

setup(name='qddynamics',
      version='1.0',
      description='Package to analyze Quantum Dot Lasing Dynamics',
      url='http://github.com/phys201/qd_laser_dynamics',
      author='Daniel Fernandez, Elizabeth Park, Alex Raun, and Hana Warner',
      author_email='dfernandez@g.harvard, spark3@g.harvard, raun@g.harvard.edu, hwarner@g.harvard.edu',
      license='GPLv3',
      packages=['qddynamics'],
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'matplotlib',
          'emcee',
          'nose',
          'unittest2',
          ],
      test_suite = 'nose.test',
      tests_require = ['nose'],
      )
