# NeuroLab API Endpoints Summary

## Complete List of Available Endpoints

### Core API Endpoints (main.py)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | API information and available endpoints | No |
| GET | `/health` | Health check and diagnostics | No |
| POST | `/upload` | Upload and process EEG files | No |
| POST | `/analyze` | Analyze EEG data | No |
| POST | `/calibrate` | Calibrate model with new data | No |
| GET | `/recommendations` | Get personalized recommendations | No |

### Authentication Endpoints (api/auth.py)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/auth/login` | User login, get JWT tokens | No |
| POST | `/api/auth/refresh` | Refresh access token | No |
| POST | `/api/auth/logout` | Logout and invalidate tokens | Yes |
| POST | `/api/auth/users` | Create new user (admin only) | Yes (Admin) |

### Training Endpoints (api/training.py) 🆕

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/train` | Train model with data | Yes (Admin) |
| POST | `/api/train/file` | Train model from uploaded file | Yes (Admin) |
| GET | `/api/train/status/{job_id}` | Get training job status | Yes |
| GET | `/api/train/jobs` | List training jobs | Yes |
| DELETE | `/api/train/job/{job_id}` | Delete training job | Yes (Admin) |
| POST | `/api/train/compare` | Compare multiple models | Yes (Admin) |

### Streaming Endpoints (api/streaming_endpoint.py)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/stream` | Stream real-time EEG data | Yes |
| POST | `/api/stream/clear` | Clear client stream buffer | Yes |

## Total Endpoints: 18

## Quick Reference

### Public Endpoints (No Auth)
- `GET /` - API information
- `GET /health` - Health check
- `POST /upload` - Upload EEG files
- `POST /analyze` - Analyze EEG data
- `POST /calibrate` - Calibrate model
- `GET /recommendations` - Get recommendations
- `POST /api/auth/login` - User login
- `POST /api/auth/refresh` - Refresh token

### Protected Endpoints (Auth Required)
- `POST /api/auth/logout` - User logout
- `POST /api/stream` - Stream real-time data
- `POST /api/stream/clear` - Clear stream buffer
- `GET /api/train/status/{job_id}` - Get training status
- `GET /api/train/jobs` - List training jobs

### Admin Only Endpoints
- `POST /api/auth/users` - Create user
- `POST /api/train` - Train model with data
- `POST /api/train/file` - Train model from file
- `DELETE /api/train/job/{job_id}` - Delete training job
- `POST /api/train/compare` - Compare models

## Interactive Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
