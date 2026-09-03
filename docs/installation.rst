User Installation
-----------------

.. note::

    The `mast-table` package is actively maintained, and improvements are made regularly.
    To ensure you have the latest features and bug fixes, we recommend updating to the most 
    recent version. The development version often includes the latest changes ahead of full releases.


Set Up Your Local Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To use mast-table, you need `Python 3.12` or later. Below is an example using `Python 3.12`, but you can
replace 3.12 with any supported version.

1. **Create a new Conda environment:**

.. code-block:: bash

    conda create -n mast-table-env python=3.12
    conda activate mast-table-env

2. **Install mast-table using pip:**

- To install the latest development version (includes the most recent updates and fixes):

.. code-block:: bash

    pip install git+https://github.com/spacetelescope/mast-table --upgrade

- To install the latest stable release

.. code-block:: bash

    pip install mast-table --upgrade

Setting Up a Local Development Copy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you plan to contribute to mast-table or modify the code, 
follow these steps to set up a local development copy:

1. **Fork the mast-table repository**

- Go to the `mast-table GitHub repository <https://github.com/spacetelescope/mast-table>`_
- Click the Fork button.
- This creates a copy of the repository under your GitHub account.
  
2. **Clone your fork locally.**

- Replace your-username with your GitHub username:

.. code-block:: bash

    git clone git@github.com:username/mast-table.git
    cd mast-table

3. **Set up the upstream remote**

.. code-block:: bash

    git remote add upstream git@github.com:spacetelescope/mast-table.git
    git fetch upstream main
    git fetch upstream --tags

4. **Keep your fork updated**

- Before starting new work, update your local ``main`` branch with the 
  latest changes from the upstream repository:

.. code-block:: bash

    git checkout main
    git pull upstream main



