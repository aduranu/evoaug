Contributing to EvoAug2
========================

Thank you for your interest in contributing to EvoAug2! This guide will help you get started.

Getting Started
--------------

**Prerequisites**:

* Python 3.8+
* Git
* Basic knowledge of PyTorch and genomic data

**Setup Development Environment**:

.. code-block:: bash

   # Fork and clone the repository
   git clone https://github.com/YOUR_USERNAME/evoaug2.git
   cd evoaug2
   
   # Install in development mode
   pip install -e ".[dev,docs]"
   
   # Install pre-commit hooks
   pre-commit install

Development Setup
----------------

**Install Development Dependencies**:

.. code-block:: bash

   pip install -e ".[dev,docs,examples]"

**Run Tests**:

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=evoaug --cov=utils
   
   # Run specific test file
   pytest tests/test_augment.py

**Code Quality Checks**:

.. code-block:: bash

   # Format code
   black evoaug/ utils/ tests/ examples/
   
   # Sort imports
   isort evoaug/ utils/ tests/ examples/
   
   # Lint code
   flake8 evoaug/ utils/ tests/ examples/
   
   # Type checking
   mypy evoaug/ utils/

**Build Documentation**:

.. code-block:: bash

   cd docs
   make html
   # Open _build/html/index.html in your browser

Contribution Areas
-----------------

We welcome contributions in the following areas:

**Code Improvements**:
* Bug fixes
* Performance optimizations
* New features
* Code refactoring

**Documentation**:
* API documentation improvements
* Tutorials and examples
* User guides
* Code comments

**Testing**:
* Unit tests
* Integration tests
* Performance benchmarks
* Test coverage improvements

**Examples**:
* New use case examples
* Performance optimization examples
* Integration examples with other libraries

**Infrastructure**:
* CI/CD improvements
* Development tools
* Build system improvements

Development Workflow
-------------------

**1. Create a Feature Branch**:

.. code-block:: bash

   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description

**2. Make Your Changes**:

* Write clear, well-documented code
* Follow the existing code style
* Add tests for new functionality
* Update documentation as needed

**3. Test Your Changes**:

.. code-block:: bash

   # Run tests
   pytest
   
   # Check code quality
   pre-commit run --all-files

**4. Commit Your Changes**:

.. code-block:: bash

   git add .
   git commit -m "feat: add new augmentation feature"
   
   # Use conventional commit format:
   # feat: new feature
   # fix: bug fix
   # docs: documentation changes
   # test: test additions/changes
   # refactor: code refactoring
   # style: code style changes
   # perf: performance improvements

**5. Push and Create Pull Request**:

.. code-block:: bash

   git push origin feature/your-feature-name
   # Create PR on GitHub

Code Style Guidelines
--------------------

**Python Code Style**:
* Follow PEP 8
* Use type hints
* Write docstrings in Google/NumPy format
* Keep functions focused and small
* Use meaningful variable names

**Docstring Format**:

.. code-block:: python

   def augment_sequence(sequence: torch.Tensor, 
                        augmentation: AugmentBase) -> torch.Tensor:
       """Apply augmentation to a sequence.
       
       Parameters
       ----------
       sequence : torch.Tensor
           Input sequence with shape (A, L).
       augmentation : AugmentBase
           Augmentation to apply.
           
       Returns
       -------
       torch.Tensor
           Augmented sequence with shape (A, L).
           
       Notes
       -----
       This function preserves the input sequence length L.
       """
       # Implementation here
       pass

**Import Organization**:
* Standard library imports first
* Third-party imports second
* Local imports last
* Alphabetical order within each group

**Example**:

.. code-block:: python

   import os
   import pathlib
   
   import numpy as np
   import torch
   import pytorch_lightning as pl
   
   from evoaug.augment import AugmentBase
   from evoaug.evoaug import RobustLoader

Testing Guidelines
-----------------

**Test Requirements**:
* All new code must have tests
* Maintain test coverage above 90%
* Tests should be fast and reliable
* Use descriptive test names

**Test Structure**:

.. code-block:: python

   def test_random_mutation_parameters():
       """Test RandomMutation parameter validation."""
       # Test valid parameters
       mutation = RandomMutation(mut_frac=0.05)
       assert mutation.mutate_frac == 0.05
       
       # Test invalid parameters
       with pytest.raises(ValueError):
           RandomMutation(mut_frac=-0.1)

**Test Naming Convention**:
* `test_function_name_scenario`
* `test_class_name_method_name_scenario`
* `test_edge_case_description`

Documentation Guidelines
-----------------------

**Documentation Standards**:
* All public APIs must be documented
* Include examples in docstrings
* Update user guides for new features
* Add changelog entries for significant changes

**Docstring Sections**:
* Brief description
* Parameters (with types and descriptions)
* Returns (with types and descriptions)
* Raises (if applicable)
* Notes (additional information)
* Examples (for complex functions)

**Example**:

.. code-block:: python

   class RandomMutation(AugmentBase):
       """Randomly mutate nucleotides in sequences.
       
       This augmentation introduces point mutations at random positions
       in the sequence, effectively simulating biological mutation processes.
       
       Parameters
       ----------
       mut_frac : float
           Fraction of nucleotides to mutate (0.0 to 1.0).
           
       Returns
       -------
       torch.Tensor
           Sequences with random mutations applied.
           
       Notes
       -----
       - Mutations preserve sequence length L
       - Random DNA is generated for mutated positions
       - Each sequence receives different random mutations
       
       Examples
       --------
       >>> mutation = RandomMutation(mut_frac=0.05)
       >>> augmented = mutation(sequences)
       """
       pass

Pull Request Process
-------------------

**Before Submitting**:
* Ensure all tests pass
* Check code quality with pre-commit hooks
* Update documentation as needed
* Add changelog entry

**Pull Request Template**:

.. code-block:: markdown

   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Test addition
   
   ## Testing
   - [ ] All tests pass
   - [ ] New tests added
   - [ ] Documentation builds
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Changelog updated

**Review Process**:
* At least one maintainer must approve
* All CI checks must pass
* Address review comments promptly
* Maintainers may request changes

Release Process
--------------

**Versioning**:
* Follow semantic versioning (MAJOR.MINOR.PATCH)
* Major version: breaking changes
* Minor version: new features
* Patch version: bug fixes

**Release Checklist**:
* Update version in `__init__.py`
* Update changelog
* Create release tag
* Build and upload to PyPI
* Update documentation

Getting Help
-----------

**Communication Channels**:
* GitHub Issues for bug reports
* GitHub Discussions for questions
* GitHub Pull Requests for contributions

**Resources**:
* Code of Conduct
* Development documentation
* API reference
* Examples and tutorials

**Questions**:
* Check existing issues and discussions
* Search documentation
* Ask in GitHub Discussions
* Open an issue for bugs

Thank You
---------

Thank you for contributing to EvoAug2! Your contributions help make the library better for everyone in the genomic deep learning community.

Every contribution, no matter how small, is appreciated and valued. 