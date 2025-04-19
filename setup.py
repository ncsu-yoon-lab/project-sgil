from setuptools import setup, find_packages

setup(
    name='project_sgil',
    version='0.1.0',
    packages=find_packages(exclude=['test', 'resource']),
    install_requires=[
        'opencv-python',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            # 'node = project_sgil.node:main'
        ],
    },
)
