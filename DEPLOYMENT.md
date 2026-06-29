# CLV Prediction API - Azure Deployment Guide

## Overview

This guide covers deploying the CLV Prediction API to Azure using Docker and either App Service or Container Apps.

---

## Prerequisites

- Azure account with active subscription
- Azure CLI installed: https://docs.microsoft.com/cli/azure/install-azure-cli
- Docker installed
- Git installed

---

## Option 1: Deploy to Azure App Service (Recommended for Beginners)

### Step 1: Prepare Your Environment

```bash
cd /path/to/deployed_CLV
source .venv/bin/activate
```

### Step 2: Make Deployment Scripts Executable

```bash
chmod +x deploy.sh
```

### Step 3: Login to Azure

```bash
az login
```

### Step 4: Run Deployment Script

```bash
./deploy.sh
```

The script will:
1. Create a resource group
2. Create an Azure Container Registry (ACR)
3. Build your Docker image
4. Push it to ACR
5. Create an App Service Plan
6. Create a Web App
7. Deploy the containerized API

### Step 5: Verify Deployment

After deployment completes (this takes ~5-10 minutes), you'll see:

```
Web App URL: https://<your-app-name>.azurewebsites.net
Health Check: https://<your-app-name>.azurewebsites.net/health
API Docs: https://<your-app-name>.azurewebsites.net/docs
```

Test the health endpoint:

```bash
curl https://<your-app-name>.azurewebsites.net/health
```

---

## Option 2: Deploy to Azure Container Apps (Modern Serverless)

### Step 1: Make Deployment Script Executable

```bash
chmod +x deploy-azure-cli.sh
```

### Step 2: Run Deployment Script

```bash
./deploy-azure-cli.sh
```

This will:
1. Prompt you to login to Azure
2. Create a resource group
3. Create an ACR
4. Build and push the Docker image
5. Create a Container Apps Environment
6. Deploy the container app with auto-scaling

### Step 3: Access Your API

Once deployed, the script will output:

```
Container App URL: https://<your-container-app>.azurecontainerapps.io
```

---

## Testing the API Locally

### Step 1: Start the API

```bash
python scripts/run_api.py
```

### Step 2: Run Test Suite

In another terminal:

```bash
python scripts/local_test.py
```

This will test all endpoints and verify the API is working correctly.

---

## Manual Deployment (Step by Step)

If you prefer to deploy manually:

```bash
# 1. Create resource group
az group create --name clv-rg --location eastus

# 2. Create ACR
az acr create --resource-group clv-rg --name clvregistry --sku Basic

# 3. Login to ACR
az acr login --name clvregistry

# 4. Build Docker image
docker build -t clv-prediction .

# 5. Tag image
docker tag clv-prediction clvregistry.azurecr.io/clv-prediction:latest

# 6. Push to ACR
docker push clvregistry.azurecr.io/clv-prediction:latest

# 7. Create App Service Plan
az appservice plan create --name clv-plan --resource-group clv-rg --is-linux --sku B2

# 8. Create Web App
az webapp create --resource-group clv-rg --plan clv-plan --name clv-webapp \
  --deployment-container-image-name clvregistry.azurecr.io/clv-prediction:latest

# 9. Configure container
az webapp config container set --name clv-webapp --resource-group clv-rg \
  --docker-custom-image-name clvregistry.azurecr.io/clv-prediction:latest \
  --docker-registry-server-url https://clvregistry.azurecr.io \
  --docker-registry-server-username $(az acr credential show --name clvregistry --query username -o tsv) \
  --docker-registry-server-password $(az acr credential show --name clvregistry --query 'passwords[0].value' -o tsv)
```

---

## API Endpoints

Once deployed, you can call the API endpoints:

### Health Check

```bash
curl https://<your-app>.azurewebsites.net/health
```

### API Documentation (Swagger UI)

```
https://<your-app>.azurewebsites.net/docs
```

### Predict CLV

```bash
curl -X POST https://<your-app>.azurewebsites.net/predict/clv \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "customer_id": "C001",
        "recency": 30,
        "frequency": 5,
        "monetary": 200.0
      }
    ]
  }'
```

### Get Recommendations

```bash
curl -X POST https://<your-app>.azurewebsites.net/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C001",
    "clv_segment": "Loyal",
    "churn_risk": "Low"
  }'
```

### Get Segment Summary

```bash
curl https://<your-app>.azurewebsites.net/segments/summary
```

### Get Top Priority Customers

```bash
curl https://<your-app>.azurewebsites.net/customers/top/10
```

---

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Then configure for your environment:

```
ENVIRONMENT=production
LOG_LEVEL=INFO
CONFIG_PATH=config.yaml
DATA_SOURCE=csv
CSV_PATH=data/customer_segmentation.csv
```

### Update App Settings in Azure Portal

1. Go to your App Service in Azure Portal
2. Navigate to **Configuration** → **Application settings**
3. Add any environment variables needed

---

## Monitoring

### View Logs

```bash
# For App Service
az webapp log tail --name clv-webapp --resource-group clv-rg

# For Container Apps
az containerapp logs show --name clv-prediction --resource-group clv-rg
```

### Enable Application Insights (Optional)

```bash
az monitor app-insights component create \
  --app clv-insights \
  --location eastus \
  --resource-group clv-rg \
  --application-type web
```

---

## Scaling

### App Service Scaling

```bash
# Scale up the plan
az appservice plan update --name clv-plan --resource-group clv-rg --sku P1V2

# Configure auto-scale
az monitor autoscale create \
  --resource-group clv-rg \
  --resource clv-plan \
  --resource-type "Microsoft.Web/serverfarms" \
  --name clv-autoscale \
  --min-count 1 \
  --max-count 10 \
  --count 2
```

### Container Apps Scaling

Container Apps auto-scale is configured in the deployment script (min 1, max 3 replicas).

To adjust:

```bash
az containerapp update \
  --name clv-prediction \
  --resource-group clv-rg \
  --min-replicas 1 \
  --max-replicas 5
```

---

## Troubleshooting

### API Not Starting

Check logs:

```bash
# App Service
az webapp log tail --name clv-webapp --resource-group clv-rg

# Container Apps
az containerapp logs show --name clv-prediction --resource-group clv-rg
```

### Health Check Failing

1. Verify the container is running
2. Check port 8000 is exposed
3. Review startup output for errors

### Database Connection Issues

If using SQL, verify:
1. Connection string is correct in config.yaml
2. Network rules allow Azure App Service to connect
3. Database credentials are valid

---

## Clean Up (Delete Resources)

```bash
# Delete entire resource group (WARNING: deletes all resources)
az group delete --name clv-rg --yes --no-wait
```

---

## Support

For more information:
- [Azure App Service Documentation](https://docs.microsoft.com/azure/app-service/)
- [Azure Container Apps Documentation](https://docs.microsoft.com/azure/container-apps/)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
