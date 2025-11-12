# NeuroLab EEG Analysis Platform - Project Summary

## Overview
NeuroLab is a comprehensive EEG (Electroencephalogram) data analysis platform that uses machine learning to classify mental states in real-time. The system processes EEG signals to identify various mental states such as stress, calmness, and focus.

## Project Structure

```
neurolab_model/
├── api/                          # API endpoints and routing
│   ├── auth.py                   # Authentication endpoints
│   ├── real_time.py              # Real-time processing logic
│   ├── security.py               # Security utilities
│   ├── streaming_endpoint.py    # Streaming data endpoints
│   └── upload.py                 # File upload handling
├── config/                       # Configuration files
│   ├── database.py               # Database configuration
│   └── settings.py               # Application settings
├── core/                         # Core business logic
│   ├── config/                   # Core configuration
│   ├── data/                     # Data handling
│   ├── ml/                       # Machine learning
│   ├── models/                   # Model definitions
│   └── services/                 # Business services
├── data/                         # Raw EEG data files
├── models/                       # ML model implementations
│   ├── model.py                  # Main model training/evaluation
│   └── model_multichannel.py    # Multi-channel model support
├── preprocessing/                # Data preprocessing modules
│   ├── features.py               # Feature extraction
│   ├── labeling.py               # Data labeling
│   ├── load_data.py              # Data loading
│   └── preprocess.py             # Preprocessing pipeline
├── processed/                    # Processed data and trained models
│   └── trained_model.h5          # Trained model file
├── tests/                        # Test suite
│   ├── conftest.py               # Pytest configuration
│   ├── performance_test.py       # Performance benchmarks
│   ├── run_tests.py              # Test runner
│   ├── test_data_handler.py      # Data handler tests
│   ├── test_data_processing.py   # Data processing tests
│   ├── test_explanation_generator.py  # Explanation tests
│   └── test_main_api.py          # API endpoint tests
├── test_data/                    # Test datasets
│   ├── sample_eeg.csv
│   └── training_eeg.csv
├── utils/                        # Utility functions
│   ├── artifacts.py              # Artifact removal
│   ├── data_handler.py           # Data handling utilities
│   ├── database_service.py       # Database operations
│   ├── duration_calculation.py   # Duration utilities
│   ├── event_detector.py         # Event detection
│   ├── explanation_generator.py  # Explanation generation
│   ├── file_handler.py           # File operations
│   ├── filters.py                # Signal filtering
│   ├── influxdb_client.py        # InfluxDB integration
│   ├── interpretability.py       # Model interpretability
│   ├── ml_processor.py           # ML processing
│   ├── model_loading.py          # Model loading utilities
│   ├── model_manager.py          # Model management
│   ├── recommendations.py        # Recommendation engine
│   ├── security.py               # Security utilities
│   ├── session_summary.py        # Session summaries
│   └── temporal_processing.py    # Temporal analysis
├── .dockerignore                 # Docker ignore file
├── .env                          # Environment variables
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore file
├── docker-compose.yml            # Docker Compose configuration
├── Dockerfile                    # Docker image definition
├── ENDPOINTS_SUMMARY.md          # API endpoints summary
├── main.py                       # Application entry point
├── nginx.conf                    # Nginx configuration
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── setup.bat                     # Windows setup script
├── setup.sh                      # Unix setup script
└── train_model.py                # Model training script
```

## Key Features

### 1. Real-time EEG Processing
- Stream and analyze EEG data in real-time
- Adaptive windowing based on signal characteristics
- Temporal smoothing for stable predictions
- Client-specific streaming buffers

### 2. Multiple File Format Support
- CSV (Comma-Separated Values)
- EDF (European Data Format)
- BDF (BioSemi Data Format)
- JSON (JavaScript Object Notation)

### 3. Advanced Signal Processing
- Artifact removal and cleaning
- Band-pass filtering
- Feature extraction (alpha, beta, theta, delta, gamma)
- Normalization and preprocessing

### 4. Machine Learning Models
- **Original CNN-LSTM**: Basic architecture
- **Enhanced CNN-LSTM**: Improved with attention mechanisms
- **ResNet-LSTM**: Residual connections for deeper networks
- **Transformer**: Attention-based architecture

### 5. Model Interpretability
- **SHAP**: Feature importance analysis
- **LIME**: Local explanations for predictions
- **Confidence Calibration**: Temperature scaling, Platt scaling
- **Reliability Diagrams**: Calibration visualization

### 6. RESTful API
- FastAPI-powered endpoints
- JWT authentication
- Rate limiting
- Request validation
- Interactive documentation (Swagger/ReDoc)

### 7. Database Integration
- **MongoDB**: Document storage for sessions and events
- **InfluxDB**: Time-series data for EEG signals
- Personalized metrics tracking
- Historical data analysis

### 8. Security Features
- JWT token authentication
- Role-based access control (RBAC)
- Data encryption in transit
- Input validation and sanitization
- Rate limiting
- Security headers

### 9. Docker Support
- Multi-stage builds for optimized images
- Docker Compose for orchestration
- Service isolation
- Volume persistence
- Health checks

### 10. Comprehensive Testing
- Unit tests
- Integration tests
- API endpoint tests
- Performance benchmarks
- Test fixtures and mocking

## API Endpoints

### Core Endpoints (12 total)

#### Public Endpoints
1. `GET /` - API information
2. `GET /health` - Health check
3. `POST /upload` - Upload EEG files
4. `POST /analyze` - Analyze EEG data
5. `POST /calibrate` - Calibrate model
6. `GET /recommendations` - Get recommendations
7. `POST /api/auth/login` - User login
8. `POST /api/auth/refresh` - Refresh token

#### Protected Endpoints
9. `POST /api/auth/logout` - User logout
10. `POST /api/stream` - Stream real-time data
11. `POST /api/stream/clear` - Clear stream buffer

#### Admin Endpoints
12. `POST /api/auth/users` - Create user

## Technology Stack

### Backend
- **Python 3.8+**
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### Machine Learning
- **TensorFlow/Keras**: Deep learning
- **scikit-learn**: ML utilities
- **SHAP**: Model interpretability
- **LIME**: Local explanations

### Data Processing
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **SciPy**: Scientific computing
- **MNE**: EEG data processing

### Databases
- **MongoDB**: Document database
- **InfluxDB**: Time-series database

### DevOps
- **Docker**: Containerization
- **Docker Compose**: Orchestration
- **Nginx**: Reverse proxy

### Testing
- **pytest**: Testing framework
- **unittest**: Unit testing
- **pytest-asyncio**: Async testing

## Model Architecture

### Enhanced CNN-LSTM (Default)
```
Input (5 features, 1 channel)
    ↓
SeparableConv1D (32 filters) + BatchNorm + Dropout
    ↓
MaxPooling1D
    ↓
SeparableConv1D (64 filters) + BatchNorm + Dropout
    ↓
MaxPooling1D
    ↓
SeparableConv1D (128 filters) + BatchNorm + Dropout
    ↓
Bidirectional LSTM (64 units) + Multi-Head Attention
    ↓
Dense (128 units) + BatchNorm + Dropout
    ↓
Dense (3 units, softmax) - Output
```

### Supported Configurations
- **8 channels**: Minimal setup
- **16 channels**: Standard setup
- **32 channels**: High-density setup
- **64 channels**: Ultra-high-density setup

## Performance Metrics

### Expected Performance
- **Inference Time**: < 100ms per sample
- **Throughput**: > 10 samples/second
- **Model Size**: < 50MB
- **Accuracy**: > 85%
- **Memory Usage**: < 2GB

### Optimization Features
- Model caching
- Batch processing
- Streaming buffers
- Adaptive windowing
- Multi-stage Docker builds

## Deployment Options

### Local Development
```bash
# Setup
python setup.py  # or setup.bat on Windows

# Train model
python train_model.py

# Start server
python main.py
```

### Docker Deployment
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Deployment
- Use environment variables for configuration
- Enable HTTPS with SSL certificates
- Configure rate limiting
- Set up monitoring and logging
- Use strong secrets and passwords
- Enable database backups

## Configuration

### Environment Variables
```bash
# MongoDB
MONGODB_URL=mongodb://mongodb:27017
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=secure_password

# InfluxDB
INFLUXDB_URL=http://influxdb:8086
INFLUXDB_TOKEN=your_token
INFLUXDB_ORG=neurolab
INFLUXDB_BUCKET=eeg_data

# Security
SECRET_KEY=your_secret_key
JWT_SECRET_KEY=your_jwt_secret
```

## Testing

### Run All Tests
```bash
# Using pytest
pytest tests/ -v

# Using unittest
python tests/run_tests.py

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Performance Benchmark
```bash
python tests/performance_test.py
```

## Documentation

- **README.md**: Project overview and setup
- **ENDPOINTS_SUMMARY.md**: API endpoints reference
- **PROJECT_SUMMARY.md**: This file
- **DOCKER.md**: Docker deployment guide
- **tests/README.md**: Testing documentation
- **Interactive Docs**: http://localhost:8000/docs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

- **GitHub**: https://github.com/your-org/neurolab_model
- **Email**: support@neurolab.cc
- **Documentation**: https://docs.neurolab.cc

## Version History

- **v1.0.0** (Current)
  - Initial release
  - Core API endpoints
  - Model training and evaluation
  - Docker support
  - Comprehensive testing
  - Authentication and security
  - Real-time streaming
  - Model interpretability

## Future Enhancements

- [ ] WebSocket support for real-time streaming
- [ ] Additional model architectures
- [ ] Advanced visualization dashboard
- [ ] Mobile app integration
- [ ] Cloud deployment templates
- [ ] Automated model retraining
- [ ] Multi-language support
- [ ] Enhanced monitoring and alerting
