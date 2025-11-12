# NeuroLab Test Suite

This directory contains comprehensive tests for the NeuroLab EEG Analysis platform.

## Test Structure

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── run_tests.py                   # Test runner script
├── test_main_api.py              # API endpoint tests
├── test_data_processing.py       # Data processing tests
├── test_data_handler.py          # Data handler tests
├── test_explanation_generator.py # Explanation generator tests
├── performance_test.py           # Model performance benchmarks
└── README.md                     # This file
```

## Running Tests

### Run All Tests
```bash
# Using unittest
python tests/run_tests.py

# Using pytest
pytest tests/

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Run Specific Test Files
```bash
# API tests
python -m unittest tests.test_main_api

# Data processing tests
python -m unittest tests.test_data_processing

# Performance benchmarks
python tests/performance_test.py
```

### Run Tests in Docker
```bash
# Build and run tests in container
docker-compose run --rm neurolab-api python tests/run_tests.py
```

## Test Categories

### 1. API Tests (`test_main_api.py`)
- Health check endpoint
- File upload and processing
- Real-time data analysis
- Model calibration
- Recommendations endpoint
- Error handling

### 2. Data Processing Tests (`test_data_processing.py`)
- CSV/JSON data loading
- Data point processing
- Buffer management
- Data validation
- Explanation generation

### 3. Data Handler Tests (`test_data_handler.py`)
- Manual data handling
- Streaming data handling
- Data saving (CSV/JSON)
- Async processing

### 4. Explanation Generator Tests (`test_explanation_generator.py`)
- Clinical observation generation
- Technical analysis
- Recommendation generation
- Context integration

### 5. Performance Tests (`performance_test.py`)
- Model inference speed
- Throughput measurement
- Memory usage
- Model size analysis
- Accuracy metrics

## Test Fixtures

The `conftest.py` file provides shared fixtures:
- `tded
E if neethis READM4. Update  80%
erage aboveintain covMa
3. ts passsure all tesch)
2. EnoapprD airst (TD frite testsres:
1. W featunewng addiing

When ntribut

## Cob
   ```xdgodb influmon up -d -compose   dockers
ceed serviart requirh
   # St  ```basrs**
 tion ErroConnecbase 

3. **Datapy
   ```train_model.n ho
   pytel firstTrain a mod# `bash
      ``und**
l Not FoMode**`

2. "
   ``}:$(pwd)YTHONPATH="${Pt PYTHONPATH   expor path
ory is innt directe pare# Ensursh
      ```barrors**
ort E

1. **Impuesn Iss

### Commoootingoublesh%

## Trcy: > 85 AccuraMB
-l size: < 50nd
- Modes/secoplesamt: > 10 ughpu Throer sample
-: < 100ms pference timeInetrics:
- nce mrmaerfod pxpecte
Ehmarks
ncormance Be
## Perf%
00endpoints: 1I > 95%
- APl paths: ritica
- C 80% coverage: >

- Overallrage Goals# Covely runs

#cheduled dai
- Smain branchto  Commits equests
-
- Pull run on:atically rautom
Tests are on
ratiIntegs uouin

## Cont
```       passt"""
 tesan up after    """Cle     lf):
(se tearDownef 
    d
   esult)tNone(rsertIsNoelf.as        sa)
dature(self.atess_feoc = pr result    "
   on""descriptit    """Tesf):
     re(seleatuest_f 
    def t_data()
   _eega = sampledat  self.""
       fixtures"est up t """Set     :
  p(self)  def setUtCase):
  t.Tesre(unitteswFeatuss TestNea

cladateeg_ort sample_est impests.conftt
from tt unittes
impor```pythonructure
t Stple TesExam
### s
w Test Writing Ne##n

ioonfiguratongoDB cMock Monfig`: k_mongodb_ction
- `mocfigurauxDB con`: Mock Inflonfiginfluxdb_cy
- `mock_onartie dicle featurampatures`: S `sample_few model
-orFloensel`: Mock Tmodmock_
- `eJSON filon`: Sample e_eeg_js`samplV file
- e CSplg_csv`: Samample_eedata
- `s EEG : Generateda`e_eeg_data
- `samplator test drectory fTemporary didir`: est_data_