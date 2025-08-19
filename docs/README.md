# EvoAug2 Documentation

This directory contains the source files for the EvoAug2 documentation, built using Sphinx and hosted on Read the Docs.

## Overview

The documentation is organized into several sections:

- **Getting Started**: Installation, quick start guide, and examples
- **User Guide**: Detailed usage instructions and best practices
- **API Reference**: Complete API documentation for all classes and functions
- **Examples**: Working examples and tutorials
- **Advanced Topics**: Architecture details and performance optimization
- **Development**: Contributing guidelines and development information

## Building Locally

### Prerequisites

1. Install Python 3.8 or higher
2. Install documentation dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Build

```bash
# Navigate to docs directory
cd docs

# Build HTML documentation
make html

# Open in browser
make open  # macOS
make open-linux  # Linux
make open-windows  # Windows
```

### Available Make Targets

- `make html` - Build HTML documentation
- `make pdf` - Build PDF documentation (requires LaTeX)
- `make clean` - Clean build directory
- `make serve` - Serve documentation locally on http://localhost:8000
- `make linkcheck` - Check for broken links
- `make help` - Show all available targets

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation index
├── Makefile             # Build automation
├── requirements.txt     # Documentation dependencies
├── installation.rst     # Installation guide
├── quickstart.rst       # Quick start guide
├── examples.rst         # Examples overview
├── examples/            # Example documentation
│   ├── lightning_module.rst
│   └── vanilla_pytorch.rst
├── api/                 # API reference
│   └── evoaug.rst
├── user_guide/          # User guide sections
├── advanced/            # Advanced topics
└── development/         # Development information
```

## Contributing to Documentation

### Adding New Content

1. **Create new RST files** in the appropriate directory
2. **Update the index.rst** to include new pages in the table of contents
3. **Follow the existing style** and formatting conventions
4. **Test locally** before submitting changes

### Documentation Style Guide

- Use **Google-style docstrings** for Python code
- Follow **Sphinx RST syntax** for documentation files
- Include **code examples** where appropriate
- Use **cross-references** to link related content
- Maintain **consistent formatting** throughout

### Code Examples

- Include **complete, runnable examples**
- Use **realistic but simple data**
- Add **explanatory comments**
- Test examples **before committing**

### Building and Testing

```bash
# Build documentation
make html

# Check for warnings
make html SPHINXOPTS="-W"

# Check links
make linkcheck

# Clean and rebuild
make clean && make html
```

## Read the Docs Integration

The documentation is automatically built and deployed on Read the Docs:

- **Project URL**: https://evoaug2.readthedocs.io/
- **Build Status**: [![Documentation Status](https://readthedocs.org/projects/evoaug2/badge/?version=latest)](https://evoaug2.readthedocs.io/en/latest/?badge=latest)

### Configuration

The `.readthedocs.yml` file in the root directory configures the Read the Docs build process:

- Python version: 3.11
- Build system: Sphinx
- Dependencies: Installed from requirements.txt
- Auto-deployment: On every push to main branch

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the package is installed in editable mode
2. **Build Failures**: Check that all dependencies are installed
3. **Missing Modules**: Verify that autodoc can find all modules
4. **Theme Issues**: Ensure sphinx-rtd-theme is properly installed

### Getting Help

- Check the [Sphinx documentation](https://www.sphinx-doc.org/)
- Review [Read the Docs documentation](https://docs.readthedocs.io/)
- Open an issue on the [GitHub repository](https://github.com/aduranu/evoaug)

## Version Information

- **Current Version**: 2.0.3
- **Documentation Version**: Latest
- **Python Support**: 3.8+
- **PyTorch Support**: 1.9.0+

## License

The documentation is licensed under the same terms as the EvoAug2 package (MIT License). 