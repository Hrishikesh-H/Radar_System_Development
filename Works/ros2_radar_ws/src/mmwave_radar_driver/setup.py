from setuptools import find_packages, setup

package_name = 'mmwave_radar_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/python3.10/site-packages/' + package_name, ['mmwave_radar_driver/best_res_4cm.cfg']),
    ],
    install_requires=['setuptools', 'pyserial', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='Hrishikesh Hiremath',
    maintainer_email='me220003037@iiti.ac.in',
    description='This is a robust ROS 2 plugin developed for Out of the box firmware on mmWave radar IWP6843 AoP 60Hz',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'radar_parser = mmwave_radar_driver.radar_parser_node:main',
            'radar_pc = mmwave_radar_driver.radar_pc:main'
        ],
    },
)
