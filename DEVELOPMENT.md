# FX AI-Quant Trading System - Development Guide

## Quick Start

### Prerequisites
- Python 3.11 or higher
- Git
- Docker (optional, for containerized development)

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Fx_Quant_System
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp env.template .env
   # Edit .env with your API keys and configuration
   ```

5. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Development Workflow

### Test-Driven Development (TDD)

This project follows TDD principles:

1. **Write tests first** - Define expected behavior before implementation
2. **Run tests** - Ensure they fail initially (red)
3. **Implement code** - Write minimal code to pass tests (green)
4. **Refactor** - Improve code while keeping tests passing (refactor)

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=core --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_core_interfaces.py -v

# Run tests in watch mode (install pytest-watch first)
ptw tests/
```

### Code Quality

#### Formatting
```bash
# Format code with Black
black .

# Check formatting
black --check .
```

#### Linting
```bash
# Run flake8 linting
flake8 . --exclude=venv

# Run with specific rules
flake8 . --select=E9,F63,F7,F82 --exclude=venv
```

#### Security Scanning
```bash
# Run security scan with Bandit
bandit -r core/ --severity-level medium

# Generate detailed report
bandit -r core/ -f json -o bandit-report.json
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with newline
- **check-yaml/json/toml**: Validates file formats
- **black**: Code formatting
- **flake8**: Linting
- **bandit**: Security scanning
- **mypy**: Type checking

## Project Structure

```
Fx_Quant_System/
├── core/                          # Core system modules
│   ├── interfaces/               # Abstract base classes
│   │   ├── data_interfaces.py   # Market data interfaces
│   │   ├── ml_interfaces.py     # ML model interfaces
│   │   ├── trading_interfaces.py # Trading system interfaces
│   │   └── messaging_interfaces.py # Message bus interfaces
│   ├── config/                   # Configuration management
│   │   └── settings.py          # Pydantic settings models
│   └── messaging/               # Message bus implementation
├── models/                       # ML models and algorithms
│   ├── regime/                  # Market regime detection
│   ├── ml/                      # Machine learning models
│   └── ensemble/                # Ensemble methods
├── strategies/                   # Trading strategies
│   ├── scalping/                # High-frequency strategies
│   ├── breakout/                # Breakout strategies
│   └── arbitrage/               # Arbitrage strategies
├── execution/                    # Order execution system
│   ├── order_management/        # Order management
│   ├── brokers/                 # Broker integrations
│   └── risk_management/         # Risk management
├── data/                        # Data storage
│   ├── raw/                     # Raw market data
│   ├── processed/               # Processed features
│   └── historical/              # Historical data
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── performance/             # Performance tests
├── monitoring/                  # System monitoring
│   ├── metrics/                 # Performance metrics
│   └── dashboard/               # Monitoring dashboard
└── scripts/                     # Utility scripts
```

## Architecture Principles

### 1. Modular Design
- Each component has a single responsibility
- Clear interfaces between modules
- Easy to test and maintain

### 2. Async/Await Pattern
- Non-blocking I/O operations
- Efficient handling of market data streams
- Concurrent processing of multiple strategies

### 3. Message-Driven Architecture
- Loose coupling between components
- Event-driven communication
- Scalable and resilient system

### 4. Configuration Management
- Environment-based configuration
- Type-safe settings with Pydantic
- Centralized configuration management

## Development Guidelines

### Code Style

1. **Follow PEP 8** - Python style guide
2. **Use type hints** - Improve code clarity and IDE support
3. **Write docstrings** - Document all public functions and classes
4. **Keep functions small** - Single responsibility principle
5. **Use meaningful names** - Self-documenting code

### Testing Guidelines

1. **Test coverage** - Aim for >90% test coverage
2. **Test isolation** - Each test should be independent
3. **Mock external dependencies** - Use unittest.mock for external services
4. **Test edge cases** - Include boundary conditions and error cases
5. **Performance tests** - Test latency-critical components

### Git Workflow

1. **Feature branches** - Create branches for new features
2. **Descriptive commits** - Use conventional commit messages
3. **Pull requests** - Code review before merging
4. **CI/CD** - Automated testing and deployment

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## Performance Considerations

### Latency Optimization

1. **Minimize allocations** - Reuse objects where possible
2. **Use efficient data structures** - NumPy arrays for numerical data
3. **Profile critical paths** - Use cProfile for performance analysis
4. **Async I/O** - Non-blocking operations for network calls

### Memory Management

1. **Monitor memory usage** - Use memory profilers
2. **Clean up resources** - Proper context managers
3. **Limit data retention** - Implement data rotation policies
4. **Use generators** - For large data processing

## Debugging

### Logging

```python
import structlog

logger = structlog.get_logger(__name__)

# Structured logging
logger.info("Order executed",
           symbol="EURUSD",
           quantity=100000,
           price=1.0851)
```

### Debugging Tools

1. **pdb** - Python debugger
2. **pytest-pdb** - Drop into debugger on test failures
3. **memory_profiler** - Memory usage analysis
4. **py-spy** - Production profiling

## Deployment

### Docker Development

```bash
# Build development image
docker build -t fx-trading-dev .

# Run with development settings
docker-compose -f docker-compose.dev.yml up

# Run tests in container
docker run --rm fx-trading-dev pytest tests/
```

### Environment Variables

Required environment variables (see `env.template`):

- **API Keys**: OANDA_API_KEY, PERPLEXITY_API_KEY, etc.
- **Database**: DATABASE_URL, REDIS_URL
- **Monitoring**: PROMETHEUS_PORT, GRAFANA_PORT
- **System**: ENVIRONMENT, DEBUG, LOG_LEVEL

## Troubleshooting

### Common Issues

1. **Import errors** - Check PYTHONPATH and virtual environment
2. **Test failures** - Ensure test database is clean
3. **Performance issues** - Profile and optimize hot paths
4. **Memory leaks** - Use memory profilers to identify issues

### Getting Help

1. **Documentation** - Check README.md and inline docs
2. **Tests** - Look at test cases for usage examples
3. **Issues** - Create GitHub issues for bugs
4. **Discussions** - Use GitHub discussions for questions

## Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Write tests for new functionality**
4. **Ensure all tests pass**
5. **Submit a pull request**

### Code Review Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] No security vulnerabilities
- [ ] Performance impact considered
- [ ] Backward compatibility maintained

## Resources

- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Type Hints (PEP 484)](https://www.python.org/dev/peps/pep-0484/)
- [Async/Await (PEP 492)](https://www.python.org/dev/peps/pep-0492/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Flake8 Linter](https://flake8.pycqa.org/)
- [Pre-commit Hooks](https://pre-commit.com/)
