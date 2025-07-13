# Installation

## Requirements

**Python Version**: 3.9 or higher

**Operating Systems**: Windows, macOS, Linux

## Installation Methods

### Via pip (Recommended)

```bash
pip install cognito-sim-engine
```

### Development Installation

For developers who want to contribute or modify the engine:

```bash
# Clone the repository
git clone https://github.com/yourusername/cognito-sim-engine.git
cd cognito-sim-engine

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### From Source

```bash
# Download the latest release
wget https://github.com/yourusername/cognito-sim-engine/archive/main.zip
unzip main.zip
cd cognito-sim-engine-main

# Install
pip install .
```

## Verify Installation

After installation, verify that everything works correctly:

```bash
# Check CLI is available
cognito-sim --version

# Run basic tests
python -c "from cognito_sim_engine import CognitiveEngine; print('✓ Installation successful')"
```

## Optional Dependencies

For enhanced functionality, consider installing these optional packages:

### Visualization

```bash
pip install matplotlib seaborn plotly
```

### Advanced Analytics

```bash
pip install pandas numpy scipy
```

### Jupyter Notebook Support

```bash
pip install jupyter ipywidgets
```

### Documentation Building

```bash
pip install mkdocs mkdocs-material mkdocstrings
```

## Troubleshooting

### Common Issues

**Import Error**: If you encounter import errors, ensure you have Python 3.9+ and all dependencies are properly installed.

**Memory Warnings**: For large simulations, ensure adequate system memory (4GB+ recommended).

**Visualization Issues**: If matplotlib plots don't display, check your backend configuration:

```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

### Getting Help

- 📚 [Documentation](../index.md)
- 🐛 [Issue Tracker](https://github.com/yourusername/cognito-sim-engine/issues)
- 💬 [Discussions](https://github.com/yourusername/cognito-sim-engine/discussions)

## Next Steps

Once installed, continue to the [Quick Start](quickstart.md) guide to create your first cognitive simulation.
