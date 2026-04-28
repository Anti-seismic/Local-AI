#!/usr/bin/env bash
# =========================================================================
# local_ai.sh
# Create the structure
# Install dependencies
# Build or clean then rebuild the "Qwen3.5-2B-AWQ-4bit" virtual environment
# =========================================================================

set -euo pipefail

echo "Installing prerequisites..."

# Install OS dependencies
sudo apt update
sudo apt upgrade -y
sudo apt install -y linux-headers-generic build-essential python3-venv python3-pip git tmux curl

curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the structure of the project
mkdir -p \
  ~/LocalAI/models/info \
  ~/LocalAI/models/Qwen3.5-2B-AWQ-4bit \
  ~/LocalAI/virtual_Env/Qwen3.5-2B-AWQ-4bit \
  ~/LocalAI/virtual_Env/ProjectUI \
  ~/LocalAI/data \
  ~/LocalAI/logs

# Install GPU drivers and toolkits
echo "Installing CUDA repo keyring..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

echo "Installing CUDA toolkit..."
sudo apt install -y cuda-toolkit-12-9

echo "Potential reboot needed there. Let's try to finish without rebooting"

echo "Installing NVIDIA driver..."
sudo ubuntu-drivers autoinstall

nvidia-smi || true
nvcc -V || true

echo "Running project setup..."
./project_env.sh

# Create the virtual environment hosting VLLM
echo "Creating venv..."
uv venv ~/LocalAI/virtual_Env/Qwen3.5-2B-AWQ-4bit --python 3.12 --seed --managed-python
source ~/LocalAI/virtual_Env/Qwen3.5-2B-AWQ-4bit/bin/activate

echo "Installing vLLM..."
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly

echo "Local AI installed."

# Instructions
echo "Now reboot,  then execute .~/LocalAI/run_ai.sh"
echo "place the files as follow:
📁 ~/LocalAI/                               ← Project Root
├── run_ai.sh                               ← Orchestrator
├── vllm_launcher.py                        ← vLLM Server Launcher
├── upload_server.py                        ← File Upload Server Launcher (with Safty and Security measures)
├── app.js                                  ← Frontend Logic ( preserves CORS and marked.parse() )
├── chat.html                               ← UI, File Picker, Tools Selector
├── style.css                               ← UI styling
├── tray.py                                 ← System tray (unchanged)
├── debug_ai.sh                             ← Cleans (./debug_ai.sh --kill) and Diagnosis (./debug_ai.sh)
├── run_ai.show                             ← Begins the Show
├── favicon.ico                             ← Browser tab icon image
├── favicon.svg                             ← Browser tab icon image
├── LocalAI_icon_256.png                    ← Desktop icon image
├── LocalAI_icon_48.png                     ← Desktop icon image
├── LocalAI.desktop                         ← Desktop icon (must be on Desktop too)
📁 ~/LocalAI/models/info/                   ← Repository: VLLM Launching Commands of the Local AI Models in ~/LocalAI/models
├── Qwen3.5-2B-AWQ-4bit.json                ← AI Models VLLM Launching Commands
📁 ~/LocalAI/models/                        ← Repository: Local AI Models
├── ~/LocalAI/models/Qwen3.5-2B-AWQ-4bit/   ← AI Model Qwen3.5-2B-AWQ-4bit
|
📁 ~/LocalAI/virtual_Env/                   ← Repository: Virtual Environments
├── /Qwen3.5-2B-AWQ-4bit/                   ← VLLM Virtual Environment for Qwen3.5-2B-AWQ-4bit Local AI
├── /ProjectUI/                             ← VLLM Virtual Environment for the GUI and the Tools supporting the AI
📁 ~/LocalAI/vllm/                          ← Git clone VLLM
|
📁 ~/LocalAI/data/                          ← Repository: Users Coversations
|
📁 ~/LocalAI/logs/                          ← Repository: Services logs
|
📁 /home/ai-broker/Desktop/
└── LocalAI.desktop                         ← Desktop icon (must be in ~/LocalAI/ too)"
echo "Then execute .~/LocalAI/run_ai.sh"
