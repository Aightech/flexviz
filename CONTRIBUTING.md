# Contributing to Flex Viewer

Thank you for your interest in contributing to Flex Viewer!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Aightech/flexviz.git
   cd flexviz
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or: venv\Scripts\activate  # Windows
   ```

3. Install development dependencies:
   ```bash
   pip install pytest pytest-cov numpy pyvista
   ```

4. Install the plugin for development:
   ```bash
   ./install.sh  # Creates symlink to KiCad plugins directory
   ```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where practical
- Keep functions focused and well-documented

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests to ensure nothing is broken
5. Commit with clear, descriptive messages
6. Push to your fork and submit a pull request

## Reporting Issues

When reporting bugs, please include:
- KiCad version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Any error messages from `flex_viewer_error.log`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
