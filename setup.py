from setuptools import setup

setup(name='robot_path_following',
      version='0.0.1',
      install_requires=['gym>=0.17.2', 'numpy>=1.18.5', 'tensorflow-gpu=1.15.0',
                        'geomdl=5.3.0', 'scipy=1.5.0', 'bezier=2020.5.19']
)
