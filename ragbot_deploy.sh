#!/bin/bash

# Deploy Modal Services and Bot App
# Run this script from the root of the pipecat-modal project
#
# Usage:
#   ./deploy_all.sh rag             # Deploy only RAG service
#   ./deploy_all.sh bot tts         # Deploy only bot and TTS services
#   ./deploy_all.sh rag stt tts bot # Deploy multiple services
#   ./deploy_all.sh --all           # Deploy all services

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default: deploy no services (selective mode)
DEPLOY_RAG=false
DEPLOY_STT=false
DEPLOY_TTS=false
DEPLOY_BOT=false

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    echo -e "${RED}❌ Error: Please specify services to deploy or use --all${NC}"
    echo ""
    echo "Usage:"
    echo "  ./deploy_all.sh rag             # Deploy only RAG service"
    echo "  ./deploy_all.sh bot tts         # Deploy bot and TTS services"
    echo "  ./deploy_all.sh --all           # Deploy all services"
    echo ""
    echo "Valid service options: rag, stt, tts, bot"
    exit 1
elif [[ "$1" == "--all" ]]; then
    # Deploy all services
    DEPLOY_RAG=true
    DEPLOY_STT=true
    DEPLOY_TTS=true
    DEPLOY_BOT=true
else
    # Parse the services to deploy
    for service in "$@"; do
        case "$service" in
            "rag")
                DEPLOY_RAG=true
                ;;
            "stt")
                DEPLOY_STT=true
                ;;
            "tts")
                DEPLOY_TTS=true
                ;;
            "bot")
                DEPLOY_BOT=true
                ;;
            *)
                echo -e "${RED}❌ Error: Unknown service '$service'${NC}"
                echo "Valid options: rag, stt, tts, bot, --all"
                exit 1
                ;;
        esac
    done
fi

# Function to deploy a service
deploy_service() {
    local service_name=$1
    local module_path=$2
    
    echo -e "\n${BLUE}📦 Deploying $service_name...${NC}"
    echo "Module: $module_path"
    
    if uv run modal deploy -m "$module_path"; then
        echo -e "${GREEN}✅ $service_name deployed successfully!${NC}"
    else
        echo -e "${RED}❌ Failed to deploy $service_name${NC}"
        exit 1
    fi
}

# Show what will be deployed
echo "🚀 Modal Services Deployment"
echo "=============================================="
services_to_deploy=()
if [[ "$DEPLOY_RAG" == "true" ]]; then services_to_deploy+=("RAG"); fi
if [[ "$DEPLOY_STT" == "true" ]]; then services_to_deploy+=("STT"); fi
if [[ "$DEPLOY_TTS" == "true" ]]; then services_to_deploy+=("TTS"); fi
if [[ "$DEPLOY_BOT" == "true" ]]; then services_to_deploy+=("Bot"); fi

if [[ ${#services_to_deploy[@]} -eq 0 ]]; then
    echo -e "${RED}❌ No services selected for deployment${NC}"
    exit 1
fi

echo -e "${YELLOW}Services to deploy: ${services_to_deploy[*]}${NC}"
echo -e "${BLUE}Starting deployment process...${NC}"

# Deploy services in order (only if enabled)
deployed_count=0

if [[ "$DEPLOY_RAG" == "true" ]]; then
    deploy_service "RAG Service" "server.services.modal_rag.vllm_rag_server"
    ((deployed_count++))
fi

if [[ "$DEPLOY_STT" == "true" ]]; then
    deploy_service "STT Service" "server.services.stt.kyutai_stt"
    ((deployed_count++))
fi

if [[ "$DEPLOY_TTS" == "true" ]]; then
    deploy_service "TTS Service" "server.services.tts.chatterbox_tts"
    ((deployed_count++))
fi

if [[ "$DEPLOY_BOT" == "true" ]]; then
    deploy_service "Bot App" "server.app"
    ((deployed_count++))
fi

echo -e "\n${GREEN}🎉 Successfully deployed $deployed_count service(s)!${NC}"
echo "=============================================="
echo -e "${BLUE}Your Modal apps are now live and ready to use.${NC}"
echo ""
echo "Example usage:"
echo "  ./deploy_all.sh rag                # Deploy only RAG service"  
echo "  ./deploy_all.sh bot tts            # Deploy bot and TTS services"
echo "  ./deploy_all.sh --all              # Deploy all services" 
