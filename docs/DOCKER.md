# Docker Deployment Guide

This guide explains how to deploy the NeuroLab EEG Analysis platform using Docker.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available
- 10GB free disk space

## Quick Start

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/neurolab-0x/ai.neurolab.git neurolab_model
cd neurolab_model

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 2. Build and Start Services

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

### 3. Verify Deployment

```bash
# Check API health
curl http://localhost:8000/health

# Check MongoDB
docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"

# Check InfluxDB
curl http://localhost:8086/health
```

## Services

### NeuroLab API
- **Port**: 8000
- **Health**: http://localhost:8000/health
- **Docs**: http://localhost:8000/docs

### MongoDB
- **Port**: 27017
- **Default credentials**: admin/admin123 (change in production!)

### InfluxDB
- **Port**: 8086
- **UI**: http://localhost:8086
- **Default credentials**: admin/admin123456 (change in production!)

### Nginx (Optional)
- **HTTP Port**: 80
- **HTTPS Port**: 443

## Configuration

### Environment Variables

Edit `.env` file:

```bash
# MongoDB
MONGODB_URL=mongodb://mongodb:27017
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=your-secure-password

# InfluxDB
INFLUXDB_URL=http://influxdb:8086
INFLUXDB_TOKEN=your-super-secret-token
INFLUXDB_ORG=neurolab
INFLUXDB_BUCKET=eeg_data

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key
```

### Volume Mounts

Data is persisted in Docker volumes:
- `mongodb-data`: MongoDB database files
- `influxdb-data`: InfluxDB time-series data
- `./data`: EEG data files
- `./processed`: Trained models
- `./logs`: Application logs

## Common Commands

### Service Management

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart a service
docker-compose restart neurolab-api

# View logs
docker-compose logs -f neurolab-api

# Execute command in container
docker-compose exec neurolab-api bash
```

### Database Operations

```bash
# MongoDB shell
docker-compose exec mongodb mongosh

# InfluxDB CLI
docker-compose exec influxdb influx

# Backup MongoDB
docker-compose exec mongodb mongodump --out /data/backup

# Backup InfluxDB
docker-compose exec influxdb influx backup /var/lib/influxdb2/backup
```

### Model Training

```bash
# Train model in container
docker-compose exec neurolab-api python train_model.py

# Run tests
docker-compose exec neurolab-api python tests/run_tests.py

# Performance benchmark
docker-compose exec neurolab-api python tests/performance_test.py
```

## Production Deployment

### 1. Security Hardening

```bash
# Generate strong secrets
openssl rand -hex 32  # For SECRET_KEY
openssl rand -hex 32  # For JWT_SECRET_KEY
openssl rand -hex 32  # For INFLUXDB_TOKEN
```

### 2. Enable HTTPS

```bash
# Generate SSL certificates (or use Let's Encrypt)
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem

# Uncomment HTTPS server block in nginx.conf
```

### 3. Resource Limits

Add to `docker-compose.yml`:

```yaml
services:
  neurolab-api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### 4. Monitoring

```bash
# Add Prometheus and Grafana
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

## Troubleshooting

### API Not Starting

```bash
# Check logs
docker-compose logs neurolab-api

# Check if model exists
docker-compose exec neurolab-api ls -la processed/

# Rebuild image
docker-compose build --no-cache neurolab-api
```

### Database Connection Issues

```bash
# Check if databases are running
docker-compose ps

# Test MongoDB connection
docker-compose exec neurolab-api python -c "from pymongo import MongoClient; print(MongoClient('mongodb://mongodb:27017').server_info())"

# Test InfluxDB connection
docker-compose exec neurolab-api python -c "from influxdb_client import InfluxDBClient; print(InfluxDBClient(url='http://influxdb:8086', token='test').ping())"