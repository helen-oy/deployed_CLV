#!/bin/bash

# Quick deployment script using Azure CLI
# Usage: ./deploy-azure-cli.sh

set -e

echo "CLV Prediction - Azure Container Apps Deployment"
echo "=================================================="

# Variables
PROJECT_NAME="clv-prediction"
RESOURCE_GROUP="clv-rg"
LOCATION="eastus"
ENVIRONMENT="clv-env"
ACR_NAME="clvregistry"
IMAGE_NAME="clv-prediction"

# Login to Azure
echo "Logging into Azure..."
az login --use-device-code

# Create resource group
echo "Creating resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

# Create ACR
echo "Creating Container Registry..."
az acr create --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" --sku Basic

# Login to ACR
echo "Logging into ACR..."
az acr login --name "$ACR_NAME"

# Build image in ACR
echo "Building Docker image in ACR (this may take a few minutes)..."
az acr build --registry "$ACR_NAME" \
  --image "$IMAGE_NAME:latest" .

# Create Container Apps Environment
echo "Creating Container Apps environment..."
az containerapp env create \
  --name "$ENVIRONMENT" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

# Deploy to Container Apps
echo "Deploying Container App..."
az containerapp create \
  --name "$PROJECT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENVIRONMENT" \
  --image "$ACR_NAME.azurecr.io/$IMAGE_NAME:latest" \
  --env-vars CONFIG_PATH=config.yaml LOG_LEVEL=INFO \
  --ingress external \
  --target-port 8000 \
  --min-replicas 1 \
  --max-replicas 3 \
  --registry-server "$ACR_NAME.azurecr.io" \
  --registry-username "$(az acr credential show --name $ACR_NAME --query username -o tsv)" \
  --registry-password "$(az acr credential show --name $ACR_NAME --query 'passwords[0].value' -o tsv)"

# Get the URL
CONTAINER_APP_URL=$(az containerapp show \
  --name "$PROJECT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  --output tsv)

echo ""
echo "=================================================="
echo "✅ Deployment Complete!"
echo "=================================================="
echo "Container App URL: https://$CONTAINER_APP_URL"
echo "Health Check: https://$CONTAINER_APP_URL/health"
echo "API Documentation: https://$CONTAINER_APP_URL/docs"
echo ""
echo "To view logs:"
echo "az containerapp logs show --name $PROJECT_NAME --resource-group $RESOURCE_GROUP"
echo ""
