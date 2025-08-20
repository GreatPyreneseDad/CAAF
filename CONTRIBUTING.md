# Contributing to CAAF

Thank you for your interest in contributing to the Coherence-Aware AI Framework! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in the [issue tracker](https://github.com/GreatPyreneseDad/CAAF/issues)
2. If not, create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (Python version, OS, etc.)

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python run_tests.py`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/CAAF.git
cd CAAF

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -r requirements.txt
pip install -e .
```

### Code Style

- Follow PEP 8
- Use meaningful variable names
- Add docstrings to all functions and classes
- Keep functions focused and small
- Write tests for new functionality

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage (>90%)
- Test edge cases and error conditions

### Documentation

- Update docstrings for any API changes
- Update README.md if adding new features
- Add examples for new functionality
- Update relevant documentation in `/docs`

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## Questions?

Feel free to open an issue for any questions about contributing!