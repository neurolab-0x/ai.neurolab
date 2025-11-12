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
git clone https://github.com/your-org/neurolab_model.git
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
`cc
neurolab.ort@ Email: suppc
-b.ceurola//docs.n https:ation:ocumentssues
- D_model/ineurolabr-org//you.comithubs://gttpes: h GitHub Issu:
-ions questsues andFor ist


## Suppor``

`E_CACHE=1LINILDKIT_INd-arg BUbuild --buile ker-composdocpendencies
 de

# Cacheose build docker-compILDKIT=1_BU
DOCKERter builds fasit forildK Use Bush
#```bald Cache


### Bui```
nfigured)
ready corignore (al Use .docke
#ependenciesy de unnecessard)
# Remov implemente(alreadyage builds se multi-stle
# Uficker`don

``Reductiomage Size n

### Iatiotimiz OprmancePerfo# ``

#:latest
`urolab-apish ne docker pu
    -api:latest .neurolab-r build -t    - docke
 t:cripild
  sge: bud:
  stadocker-buil

```yamlitLab CI
### G``

est
`b-api:latla push neuro      dockerst .
    atelab-api:lild -t neuror bu      docke  
  n: |ru   mage
     ush Docker i Build and pe:  - nam  
  ut@v2hecko: actions/cuses
      -     steps:test
lan: ubuntu-   runs-o
 uild:  bjobs:
 ]

mainches: [ :
    bran  push
on:
nd Deploy
ild aer Buname: Dock
```yaml
b Actions
itHu
### Gn
ratioInteg# CI/CD ``

#ore
`fluxdb2/rest/ine /var/libux restoruxdb infle exec inflomposer-c
dock/restorefluxdb2inlib/luxdb:/var/olab-inffluxdb neur/backups/iner cp .xDB
docke Influstor
# Re/restore
store /datab mongorexec mongodompose edocker-cta/restore
ngodb:/daurolab-mos/mongodb ne cp ./backupB
dockerstore MongoDRe
```bash
#  Restore

```

###dbups/influx ./back2/backupib/influxdb:/var/llab-influxdbeuro
docker cp ndbackups/mongokup ./bbac/data/b:ngodab-moer cp neurolup
dockckdb2/bab/influxp /var/lix backuxdb influ exec influker-composeckup
docout /data/bamp --dungodb mongopose exec mocker-comdoackup
 Manual bkup.sh

#bac./scripts/t
ckup scripbate bash
# Crea``

`ckup
### Ba Restore
ackup and## BG
```

memory: 8 '4'
           cpus:its:
 imrces:
    l resouy:
 
deploe.ymlr-composockeces in dase resour
# Incre```yamlScaling

l ca
### Verti
gured)
```y confieadinx alrer (ngload balancse 

# Ui=3 neurolab-apled --scampose up -
docker-co serviceScale APIh
# 

```bascalingzontal Sori

### Hngli Sca
```

##d instea 8001se"  # U8000  - "8001:ts:
se.yml
porompo docker-ce port in Changndows

#:8000  # Wi findstr tat -ano |/Mac
netsux0  # Lin -i :800t
lsoforss using pproce
# Find bashse

``` in UAlreadyort `

### Pr
``ete paramatch-size--b or pass _model.pyEdit train# raining
 size in te batch# Reduc

ces > Memoryoures> R Settings p:r DesktoDocket
# ory limiker memse Doc# Increastats

e
docker ory usag# Check mem

```bash
oryem## Out of M``

#