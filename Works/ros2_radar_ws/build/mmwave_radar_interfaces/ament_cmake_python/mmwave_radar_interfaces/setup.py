from setuptools import find_packages
from setuptools import setup

setup(
    name='mmwave_radar_interfaces',
    version='0.0.0',
    packages=find_packages(
        include=('mmwave_radar_interfaces', 'mmwave_radar_interfaces.*')),
)
