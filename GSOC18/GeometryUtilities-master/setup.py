from setuptools import setup

setup(
    name = 'geometry',
    version = '1.0.0',
    packages = ['geometry'],
    install_requires = [
        'attrs',
        'numpy',
        'scipy',
        'root-numpy',
        'shapely',
        'descartes',
        'matplotlib'
        ]
)
