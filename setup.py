from setuptools import setup, find_packages

setup(
    name = 'spamm',
    version = '0.0.1',
    description = 'AGN spectral Bayesian decomposition',
    keywords = ['astronomy'],
    classifiers = ['Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Development Status :: 1 - Planning',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Topic :: Software Development :: Libraries :: Python Modules'],
    packages = find_packages(),
    install_requires = ['setuptools',
                        'numpy',
                        'astropy',
                        'matplotlib',
                        'scipy>=0.17.1',
                        'future',
                        'emcee',
                        'pysynphot',
                        'six']
    )

