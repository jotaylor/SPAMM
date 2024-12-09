Installation
============

To install SPAMM, you first need `anaconda <https://www.anaconda.com/download/success>`_ or
`miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ installed on your system. 
You can also use `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ if you prefer. 
In this example, we'll use conda. After installing conda, create a new conda environment like so:
``conda create -n spamm python=3.11``. SPAMM has been tested with Python 3.11, but should work with later versions too.
This will create an environment named `spamm`, which you activate like so: ``conda activate spamm``. 

To install SPAMM, clone the `repository <https://github.com/oliverdamkjaer/SPAMM>`_ 
and make sure you are in the SPAMM directory. Then run ``pip install .`` to install SPAMM into your environment.

To follow the example notebooks, you'll also need to install jupyter: ``conda install jupyter-lab``. 
After it is installed, run the notebook using ``jupyter lab``. Now you're all ready to start SPAMMing!
