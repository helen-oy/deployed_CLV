#!/bin/bash

# Azure Deployment Script for CLV Prediction API
# Prerequisites: Azure CLI installed and logged in (az login)

set -e

# Configuration
PROJECT_NAME="clv-prediction"
RESOURCE_GROUP="clv-rg"
LOCATION="eastus"
ACR_NAME="clvregistry"
APP_SERVICE_PLAN="clv-plan"
WEBAPP_NAME="clv-webapp"
IMAGE_TAG="latest"

echo "=========================================="
echo "CLV Prediction API - Azure Deployment"
echo "=========================================="

# Step 1: Create Resource Group
echo -e "\n[Step 1] Creating Azure Resource Group..."
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION"

# Step 2: Create Azure Container Registry (ACR)
echo -e "\n[Step 2] Creating Azure Container Registry..."
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Basic \
  --admin-enabled true

# Step 3: Get ACR credentials
echo -e "\n[Step 3] Retrieving ACR credentials..."
ACR_URL=$(az acr show \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --query loginServer \
  --output tsv)

echo "ACR URL: $ACR_URL"

# Step 4: Build and push Docker image
echo -e "\n[Step 4] Building Docker image..."
docker build -t "$PROJECT_NAME:$IMAGE_TAG" .

echo -e "\n[Step 5] Logging into ACR..."
az acr login --name "$ACR_NAME"

echo -e "\n[Step 6] Tagging Docker image..."
docker tag "$PROJECT_NAME:$IMAGE_TAG" "$ACR_URL/$PROJECT_NAME:$IMAGE_TAG"

echo -e "\n[Step 7] Pushing Docker image to ACR..."
docker push "$ACR_URL/$PROJECT_NAME:$IMAGE_TAG"

# Step 8: Create App Service Plan
echo -e "\n[Step 8] Creating App Service Plan..."
az appservice plan create \
  --name "$APP_SERVICE_PLAN" \
  --resource-group "$RESOURCE_GROUP" \
  --is-linux \
  --sku B2

# Step 9: Create Web App
echo -e "\n[Step 9] Creating Azure Web App..."
az webapp create \
  --resource-group "$RESOURCE_GROUP" \
  --plan "$APP_SERVICE_PLAN" \
  --name "$WEBAPP_NAME" \
  --deployment-container-image-name "$ACR_URL/$PROJECT_NAME:$IMAGE_TAG"

# Step 10: Configure Web App Container Settings
echo -e "\n[Step 10] Configuring Web App container settings..."
az webapp config container set \
  --name "$WEBAPP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --docker-custom-image-name "$ACR_URL/$PROJECT_NAME:$IMAGE_TAG" \
  --docker-registry-server-url "https://$ACR_URL" \
  --docker-registry-server-username "$(az acr credential show --name $ACR_NAME --query username -o tsv)" \
  --docker-registry-server-password "$(az acr credential show --name $ACR_NAME --query 'passwords[0].value' -o tsv)"

# Step 11: Configure App Settings
echo -e "\n[Step 11] Configuring app settings..."
az webapp config appsettings set \
  --name "$WEBAPP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings \
    CONFIG_PATH=config.yaml \
    LOG_LEVEL=INFO \
    ENVIRONMENT=production \
    WEBSITES_PORT=8000

# Step 12: Enable logging
echo -e "\n[Step 12] Enabling logging..."
az webapp log config \
  --name "$WEBAPP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --docker-container-logging filesystem \
  --level information

# Get webapp URL
echo -e "\n=========================================="
echo "Deployment Complete!"
echo "=========================================="
WEBAPP_URL=$(az webapp show \
  --name "$WEBAPP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query 'defaultHostName' \
  --output tsv)

echo -e "\nWeb App URL: https://$WEBAPP_URL"
echo -e "Health Check: https://$WEBAPP_URL/health"
echo -e "API Docs: https://$WEBAPP_URL/docs"
echo ""
echo "Next steps:"
echo "1. Upload config.yaml to your Web App"
echo "2. Test the health endpoint"
echo "3. Monitor logs: az webapp log tail --name $WEBAPP_NAME --resource-group $RESOURCE_GROUP"
echo ""
