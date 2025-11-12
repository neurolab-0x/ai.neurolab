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

### Streaming Endpoints (api/streaming_endpoint.py)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/stream` | Stream real-time EEG data | Yes |
| POST | `/api/stream/clear` | Clear client stream buffer | Yes |

## Total Endpoints: 12

## Quick Reference

### Public Endpoints (No Auth)
- `GET /`
- `GET /health`
- `POST /upload`
- `POST /analyze`
- `POST /calibrate`
- `GET /recommendations`
- `POST /api/auth/login`
- `POST /api/auth/refresh`

### Protected Endpoints (Auth Required)
- `POST /api/auth/logout`
- `POST /api/stream`
- `POST /api/stream/clear`

### Admin Only Endpoints
- `POST /api/auth/users`

## Interactive Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
