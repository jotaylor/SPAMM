from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

setup(
    name='spamm',
    version='0.1',
    description='AGN spectral Bayesian decomposition',
    url='https://github.com/oliverdamkjaer/SPAMM',
    author='Oliver Damkjær',
    author_email='od.damkjaer@gmail.com',
    keywords=['astronomy'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=find_packages(),
    package_data={'spamm.utils': ['parameters.yaml']},
    install_requires=[
        'setuptools',
        'pysynphot',
        'astropy>=3.1.2',
        'dill>=0.2.9',
        'matplotlib>=3.0.3',
        'numpy>=1.16.2',
        'pyfftw>=0.9.2',
        'scipy>=1.2.1',
        'specutils>=0.5.2',
        'PyYAML',
        'emcee>=3.1.4',
        'tqdm>=4.66.1'
    ],
    python_requires='>=3.11',
)

