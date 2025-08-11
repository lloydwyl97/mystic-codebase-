# Code Quality System Documentation

## Overview

This project uses a comprehensive, professional-grade code quality system with automated checks for:

- **Code Style**: Black, isort, flake8, autopep8
- **Type Safety**: MyPy with strict settings
- **Security**: Bandit, Safety, Semgrep
- **Code Quality**: Pylint, Vulture, Radon, Lizard, Pylama
- **Testing**: Pytest with coverage reporting
- **Documentation**: Docformatter, Sphinx
- **Performance**: Memory profiler, line profiler

## Quick Start

### Running Quality Checks

```powershell
# Full comprehensive check (recommended for CI/CD)
.\run_quality_checks.ps1

# Quick check for development (faster feedback)
.\quick_check.ps1

# Individual tools
python -m black --check .
python -m isort --check-only .
python -m flake8 .
python -m mypy .
python -m bandit -r .
```

### Pre-commit Hooks (when git is available)

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black
```

## Configuration Files

- `.flake8` - Flake8 linting configuration
- `pyproject.toml` - Black and isort configuration
- `mypy.ini` - MyPy type checking configuration
- `.bandit` - Bandit security scanning configuration
- `pytest.ini` - Pytest test configuration
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `.vulture.cfg` - Vulture unused code detection
- `.radon.cfg` - Radon complexity analysis
- `.lizardrc` - Lizard code metrics
- `.pylama.ini` - Pylama static analysis

## Quality Standards

### Code Style
- **Line Length**: 100 characters
- **Formatting**: Black (uncompromising)
- **Imports**: isort with Black profile
- **Linting**: Flake8 with comprehensive rules

### Type Safety
- **MyPy**: Strict mode enabled
- **Type Coverage**: Aim for 100%
- **Type Annotations**: Required for all functions

### Security
- **Bandit**: High and medium severity issues must be fixed
- **Safety**: All vulnerable dependencies must be updated
- **Semgrep**: Custom security rules

### Code Quality
- **Cyclomatic Complexity**: Max 10 per function
- **Unused Code**: Vulture with 80% confidence
- **Code Duplication**: JSCPD detection
- **Documentation**: Docstrings for all public APIs

## CI/CD Integration

The project includes GitHub Actions workflows that run on:
- Push to main/develop branches
- Pull requests

### Workflow Features
- Automated quality checks
- Coverage reporting
- PR comments with results
- Artifact uploads

## Development Workflow

1. **Before committing**: Run `.\quick_check.ps1`
2. **Before pushing**: Run `.\run_quality_checks.ps1`
3. **For PRs**: CI will automatically run all checks

## Troubleshooting

### Common Issues

1. **MyPy errors**: Add type annotations or use `# type: ignore`
2. **Black formatting**: Run `python -m black .` to auto-format
3. **Import sorting**: Run `python -m isort .` to auto-sort
4. **Security issues**: Review and fix according to Bandit/Safety recommendations

### Performance

- Quick checks take ~30 seconds
- Full checks take ~2-3 minutes
- CI runs take ~5-10 minutes

## Customization

### Adding New Tools

1. Add to `requirements-dev.txt`
2. Create configuration file
3. Add to `run_quality_checks.py`
4. Update pre-commit config if needed

### Modifying Standards

Edit the respective configuration files:
- `.flake8` for linting rules
- `pyproject.toml` for formatting
- `mypy.ini` for type checking
- etc.

## Reports

Quality check reports are generated in:
- `quality_report.json` - Detailed JSON report
- `htmlcov/` - Coverage HTML report
- Console output - Real-time feedback

## Support

For issues with the quality system:
1. Check the tool's documentation
2. Review configuration files
3. Run individual tools for specific errors
4. Check CI logs for detailed error messages
