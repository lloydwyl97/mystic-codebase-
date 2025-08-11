# Mystic Trading Platform Test Suite

This directory contains comprehensive unit tests and integration tests for the Mystic Trading Platform backend.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_exceptions.py       # Exception handling system tests
├── test_market_data.py      # Market data module tests
├── test_database.py         # Database module tests
├── test_api_endpoints.py    # API endpoints tests
├── test_strategy_system.py  # Strategy system tests
├── test_notifications.py    # Notifications system tests
├── test_ai_modules.py       # AI modules tests
├── test_metrics.py          # Metrics and analytics tests
├── test_integration.py      # Integration tests
├── run_tests.py            # Test runner script
└── README.md               # This file
```

## Test Coverage

### Unit Tests

#### 1. Exception Handling (`test_exceptions.py`)
- **Coverage**: All exception classes, decorators, and utility functions
- **Tests**:
  - Base `MysticException` class functionality
  - Specialized exception classes (Database, API, Trading, etc.)
  - Exception decorators (`@handle_exception`, `@handle_async_exception`)
  - Safe execution functions
  - Error code mapping and HTTP exception handling
  - Exception inheritance and instantiation

#### 2. Market Data (`test_market_data.py`)
- **Coverage**: Market data fetching, processing, and caching
- **Tests**:
  - `MarketData` dataclass functionality
  - `MarketDataManager` initialization and operations
  - Live data fetching from multiple sources (Binance, Coinbase, fallback)
  - Data freshness checking and caching
  - Symbol management (add/remove)
  - Error handling for API failures
  - Market summary generation

#### 3. Database (`test_database.py`)
- **Coverage**: Database connection, operations, and error handling
- **Tests**:
  - `DatabaseManager` initialization and connection management
  - Query execution with parameters
  - Transaction handling (commit/rollback)
  - Connection error handling
  - SQL injection prevention
  - Performance testing with large datasets
  - Schema operations and constraints

#### 4. API Endpoints (`test_api_endpoints.py`)
- **Coverage**: All API endpoints and routing
- **Tests**:
  - Root and health check endpoints
  - Market data endpoints
  - Trading endpoints
  - AI endpoints
  - Analytics endpoints
  - Authentication endpoints
  - Error handling and validation
  - WebSocket connections
  - Performance testing

#### 5. Strategy System (`test_strategy_system.py`)
- **Coverage**: Strategy execution, analysis, and optimization
- **Tests**:
  - `StrategyExecutor` functionality
  - Strategy loading, updating, and deletion
  - Condition evaluation and action execution
  - `StrategyAnalyzer` performance analysis
  - Risk metrics calculation
  - Strategy validation and optimization
  - Backtesting functionality

#### 6. Notifications (`test_notifications.py`)
- **Coverage**: Alert management and message handling
- **Tests**:
  - `AlertManager` functionality
  - Alert creation, updating, and deletion
  - Alert checking and triggering
  - `MessageHandler` functionality
  - Message creation and management
  - Notification sending and caching
  - Bulk operations and cleanup

#### 7. AI Modules (`test_ai_modules.py`)
- **Coverage**: AI brains, predictions, and learning
- **Tests**:
  - `AIBrain` functionality
  - Prediction generation and market analysis
  - `AIPredictor` feature processing and model interaction
  - `AILearner` training and model management
  - Technical indicator calculation
  - Model validation and optimization
  - Performance metrics and reporting

#### 8. Metrics (`test_metrics.py`)
- **Coverage**: Metrics collection and analytics
- **Tests**:
  - `MetricsCollector` functionality
  - Performance metrics calculation
  - Market metrics collection
  - System metrics monitoring
  - `AnalyticsEngine` analysis capabilities
  - Risk metrics and portfolio analysis
  - Report generation and export

### Integration Tests (`test_integration.py`)
- **Coverage**: Module interactions and data flow
- **Tests**:
  - Market data to strategy execution flow
  - AI predictions influencing strategy decisions
  - Alert triggering and notification flow
  - Metrics collection feeding analytics
  - End-to-end trading cycle
  - Error propagation across modules
  - Performance and scalability testing
  - Configuration propagation

## Running Tests

### Prerequisites

1. **Python Environment**: Ensure you have Python 3.11+ installed
2. **Dependencies**: Install test dependencies:
   ```bash
   pip install pytest pytest-asyncio pytest-cov pytest-mock
   pip install bandit flake8 mypy
   ```

### Basic Test Execution

#### Run All Tests
```bash
cd backend
python tests/run_tests.py
```

#### Run Specific Test Files
```bash
python tests/run_tests.py --test-paths tests/test_market_data.py tests/test_database.py
```

#### Run with Coverage
```bash
python tests/run_tests.py --verbose
```

#### Run Tests in Parallel
```bash
python tests/run_tests.py --parallel
```

### Individual Test Categories

#### Security Tests Only
```bash
python tests/run_tests.py --security-only
```

#### Linting Tests Only
```bash
python tests/run_tests.py --linting-only
```

#### Type Checking Only
```bash
python tests/run_tests.py --type-only
```

### Using Pytest Directly

#### Run All Tests
```bash
pytest tests/
```

#### Run with Coverage
```bash
pytest tests/ --cov=modules --cov=utils --cov-report=html
```

#### Run Specific Test
```bash
pytest tests/test_market_data.py::TestMarketDataManager::test_get_market_data_success
```

#### Run Tests by Marker
```bash
pytest tests/ -m "unit"      # Run unit tests only
pytest tests/ -m "integration"  # Run integration tests only
pytest tests/ -m "slow"      # Run slow tests only
```

## Test Configuration

### Pytest Configuration (`conftest.py`)

The `conftest.py` file provides:
- **Fixtures**: Common test data and mock objects
- **Event Loop**: Async test support
- **Markers**: Custom test markers for categorization
- **Cleanup**: Automatic test data cleanup

### Test Data Fixtures

Common test fixtures include:
- `sample_market_data`: Market data for testing
- `sample_trades`: Trade data for testing
- `sample_strategy_config`: Strategy configuration
- `sample_alert_config`: Alert configuration
- `mock_binance_client`: Mock Binance API client
- `mock_order_manager`: Mock order management
- `temp_db_path`: Temporary database path

## Test Reports

### Coverage Report
After running tests with coverage, view the HTML report:
```bash
open htmlcov/index.html
```

### Test Report
The test runner generates a JSON report with:
- Test results and status
- Security scan results
- Linting results
- Type checking results
- Summary statistics

### Security Report
Bandit generates a security report:
```bash
cat security_report.json
```

## Writing New Tests

### Test Structure
```python
class TestNewModule:
    """Test the NewModule class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.module = NewModule()

    def test_basic_functionality(self):
        """Test basic functionality"""
        result = self.module.some_function()
        assert result == expected_value

    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async functionality"""
        result = await self.module.async_function()
        assert result == expected_value
```

### Test Guidelines

1. **Naming**: Use descriptive test names that explain what is being tested
2. **Isolation**: Each test should be independent and not rely on other tests
3. **Mocking**: Use mocks for external dependencies (APIs, databases)
4. **Coverage**: Aim for high test coverage, especially for critical paths
5. **Error Cases**: Test both success and error scenarios
6. **Async Tests**: Use `@pytest.mark.asyncio` for async functions
7. **Fixtures**: Use fixtures for common test data and setup

### Adding New Fixtures

Add new fixtures to `conftest.py`:
```python
@pytest.fixture
def new_test_data():
    """New test data fixture"""
    return {
        "key": "value",
        "data": [1, 2, 3]
    }
```

## Continuous Integration

### GitHub Actions
The test suite is configured to run automatically on:
- Pull requests
- Push to main branch
- Scheduled runs

### Local CI
Run the full CI pipeline locally:
```bash
python tests/run_tests.py --verbose --parallel
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the backend directory is in Python path
2. **Async Test Failures**: Check that `@pytest.mark.asyncio` is used
3. **Mock Issues**: Verify mock objects are properly configured
4. **Database Errors**: Use in-memory databases for testing

### Debug Mode
Run tests in debug mode:
```bash
pytest tests/ -s --pdb
```

### Verbose Output
Get detailed test output:
```bash
pytest tests/ -v -s
```

## Performance Testing

### Load Testing
```bash
pytest tests/test_integration.py::TestPerformanceIntegration -v
```

### Memory Testing
```bash
pytest tests/ --durations=10 --maxfail=1
```

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain high test coverage
4. Update this README if needed
5. Add integration tests for new module interactions

## Test Metrics

- **Total Tests**: 200+ unit tests
- **Coverage Target**: >90% code coverage
- **Test Categories**: Unit, Integration, Security, Performance
- **Execution Time**: <30 seconds for full suite
- **Success Rate**: >95% pass rate
