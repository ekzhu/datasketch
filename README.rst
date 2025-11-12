datasketch: Big Data Looks Small
================================

.. image:: https://static.pepy.tech/badge/datasketch/month
    :target: https://pepy.tech/project/datasketch

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.598238.svg
   :target: https://zenodo.org/doi/10.5281/zenodo.598238

datasketch gives you probabilistic data structures that can process and
search very large amount of data super fast, with little loss of
accuracy.

This package contains the following data sketches:

+-------------------------+-----------------------------------------------+
| Data Sketch             | Usage                                         |
+=========================+===============================================+
| `MinHash`_              | estimate Jaccard similarity and cardinality   |
+-------------------------+-----------------------------------------------+
| `Weighted MinHash`_     | estimate weighted Jaccard similarity          |
+-------------------------+-----------------------------------------------+
| `HyperLogLog`_          | estimate cardinality                          |
+-------------------------+-----------------------------------------------+
| `HyperLogLog++`_        | estimate cardinality                          |
+-------------------------+-----------------------------------------------+

The following indexes for data sketches are provided to support
sub-linear query time:

+---------------------------+-----------------------------+------------------------+
| Index                     | For Data Sketch             | Supported Query Type   |
+===========================+=============================+========================+
| `MinHash LSH`_            | MinHash, Weighted MinHash   | Jaccard Threshold      |
+---------------------------+-----------------------------+------------------------+
| `LSHBloom`_               | MinHash, Weighted MinHash   | Jaccard Threshold      |
+---------------------------+-----------------------------+------------------------+
| `MinHash LSH Forest`_     | MinHash, Weighted MinHash   | Jaccard Top-K          |
+---------------------------+-----------------------------+------------------------+
| `MinHash LSH Ensemble`_   | MinHash                     | Containment Threshold  |
+---------------------------+-----------------------------+------------------------+
| `HNSW`_                   | Any                         | Custom Metric Top-K    |
+---------------------------+-----------------------------+------------------------+

datasketch must be used with Python 3.9 or above, NumPy 1.11 or above, and Scipy.

Note that `MinHash LSH`_ and `MinHash LSH Ensemble`_ also support Redis and Cassandra 
storage layer (see `MinHash LSH at Scale`_).

Install
-------

To install datasketch using ``pip``:

.. code-block:: bash

    pip install datasketch

This will also install NumPy as dependency.

To install with Redis dependency:

.. code-block:: bash

    pip install datasketch[redis]

To install with Cassandra dependency:

.. code-block:: bash

    pip install datasketch[cassandra]

To install with Bloom filter dependency:

.. code-block:: bash

    pip install datasketch[bloom]

.. _`MinHash`: https://ekzhu.github.io/datasketch/minhash.html
.. _`Weighted MinHash`: https://ekzhu.github.io/datasketch/weightedminhash.html
.. _`HyperLogLog`: https://ekzhu.github.io/datasketch/hyperloglog.html
.. _`HyperLogLog++`: https://ekzhu.github.io/datasketch/hyperloglog.html#hyperloglog-plusplus
.. _`MinHash LSH`: https://ekzhu.github.io/datasketch/lsh.html
.. _`MinHash LSH Forest`: https://ekzhu.github.io/datasketch/lshforest.html
.. _`MinHash LSH Ensemble`: https://ekzhu.github.io/datasketch/lshensemble.html
.. _`LSHBloom`: https://ekzhu.github.io/datasketch/lshbloom.html
.. _`Minhash LSH at Scale`: http://ekzhu.github.io/datasketch/lsh.html#minhash-lsh-at-scale
.. _`HNSW`: https://ekzhu.github.io/datasketch/documentation.html#hnsw

Contributing
------------

We welcome contributions from everyone. Whether you're fixing bugs, adding features, improving documentation, or helping with tests, your contributions are valuable.

Development Setup
^^^^^^^^^^^^^^^^^

The project uses `uv` for fast and reliable Python package management. Follow these steps to set up your development environment:

1. **Install uv**: Follow the official installation guide at https://docs.astral.sh/uv/getting-started/installation/

2. **Clone the repository**:

   .. code-block:: bash

       git clone https://github.com/ekzhu/datasketch.git
       cd datasketch

3. **Set up the environment**:

   .. code-block:: bash

       # Create a virtual environment
       # (Optional: specify Python version with --python 3.x)
       uv venv
       # Activate the virtual environment (optional, uv run commands work without it)
       source .venv/bin/activate

       # Install all dependencies
       uv sync

4. **Verify installation**:

   .. code-block:: bash

       # Run tests to ensure everything works
       uv run pytest

5. **Optional dependencies** (for specific development needs):

   .. code-block:: bash

       # For testing
       uv sync --extra test

       # For Cassandra support
       uv sync --extra cassandra

       # For Redis support
       uv sync --extra redis

       # For all extras
       uv sync --all-extras

Learn more about `uv` at https://docs.astral.sh/uv/

Development Workflow
^^^^^^^^^^^^^^^^^^^^

1. **Fork the repository** on GitHub if you haven't already.

2. **Create a feature branch** for your changes:

   .. code-block:: bash

       git checkout -b feature/your-feature-name
       # Or for bug fixes:
       git checkout -b fix/issue-description

3. **Make your changes** following the project's coding standards.

4. **Run the tests** to ensure nothing is broken:

   .. code-block:: bash

       uv run pytest

5. **Check code quality** with ruff:

   .. code-block:: bash

       # Check for issues
       uvx ruff check .

       # Auto-fix formatting issues
       uvx ruff format .

6. **Commit your changes** with a clear, descriptive commit message:

   .. code-block:: bash

       git commit -m "Add feature: brief description of what was changed"

7. **Push to your fork** and create a pull request on GitHub:

   .. code-block:: bash

       git push origin your-branch-name

8. **Respond to feedback** from maintainers and iterate on your changes.

Guidelines
^^^^^^^^^^

- Follow PEP 8 style guidelines
- Write tests for new features
- Update documentation as needed
- Keep commits focused and atomic
- Be respectful in discussions

For more information, check the `GitHub issues <https://github.com/ekzhu/datasketch/issues>`_ for current priorities or areas needing help. You can also join the discussion on `project roadmap and priorities <https://github.com/ekzhu/datasketch/discussions/252>`_.