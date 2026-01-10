#!/bin/bash
# Installer for transcript-cli with NVIDIA GPU support
# Usage: curl -sSL https://raw.githubusercontent.com/aaditagrawal/transcript-cli/main/install.sh | bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "🎙️ Installing Transcript CLI..."

# Check NVIDIA
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA drivers not found. Install drivers first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Detect CUDA version
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' || echo "")

if [ -n "$NVCC_VERSION" ]; then
    CUDA_MAJOR=$(echo $NVCC_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $NVCC_VERSION | cut -d. -f2)
else
    # Fallback: check /usr/local/cuda
    if [ -d "/usr/local/cuda" ]; then
        NVCC_VERSION=$(/usr/local/cuda/bin/nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        CUDA_MAJOR=$(echo $NVCC_VERSION | cut -d. -f1)
        CUDA_MINOR=$(echo $NVCC_VERSION | cut -d. -f2)
    else
        CUDA_MAJOR=12
        CUDA_MINOR=1
    fi
fi

echo -e "${GREEN}✓${NC} CUDA: ${CUDA_MAJOR}.${CUDA_MINOR}"

# Determine PyTorch CUDA version
if [ "$CUDA_MAJOR" -ge 12 ]; then
    if [ "$CUDA_MINOR" -ge 4 ]; then
        PYTORCH_CUDA="cu124"
    else
        PYTORCH_CUDA="cu121"
    fi
elif [ "$CUDA_MAJOR" -eq 11 ]; then
    PYTORCH_CUDA="cu118"
else
    echo -e "${RED}❌ CUDA $CUDA_MAJOR.$CUDA_MINOR not supported. Need CUDA 11.8+${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} PyTorch variant: $PYTORCH_CUDA"

# Check for Blackwell GPUs (RTX 50 series) - need nightly PyTorch
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ "$GPU_NAME" == *"5070"* ]] || [[ "$GPU_NAME" == *"5080"* ]] || [[ "$GPU_NAME" == *"5090"* ]]; then
    echo -e "${YELLOW}⚠️${NC} Detected Blackwell GPU ($GPU_NAME)"
    echo "   Blackwell requires PyTorch nightly for CUDA sm_120 support"
    PYTORCH_NIGHTLY=true
else
    PYTORCH_NIGHTLY=false
fi

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}⚠️${NC} FFmpeg not found. Please install: sudo apt install ffmpeg"
    exit 1
fi

# Install uv if needed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Setup
INSTALL_DIR="$HOME/.local/share/transcript-cli"
mkdir -p "$INSTALL_DIR" "$HOME/.local/bin"
cd "$INSTALL_DIR"

uv venv --python 3.11 2>/dev/null || uv venv
source .venv/bin/activate

if [ "$PYTORCH_NIGHTLY" = true ]; then
    echo "📦 Installing PyTorch nightly (Blackwell support)..."
    uv pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
else
    echo "📦 Installing PyTorch ($PYTORCH_CUDA)..."
    uv pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$PYTORCH_CUDA"
fi

echo "📦 Installing transcript-cli..."
uv pip install git+https://github.com/aaditagrawal/transcript-cli.git
uv pip install transformers accelerate

echo "📦 Installing Flash Attention 2..."
uv pip install flash-attn --no-build-isolation 2>/dev/null || echo -e "${YELLOW}⚠️${NC} Flash Attention skipped (optional)"

# Create wrapper
cat > "$HOME/.local/bin/transcript" << 'EOF'
#!/bin/bash
source "$HOME/.local/share/transcript-cli/.venv/bin/activate"
python -m transcript_cli.main "$@"
EOF
chmod +x "$HOME/.local/bin/transcript"

# Add to PATH
grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

echo ""
echo -e "${GREEN}✅ Done!${NC} Restart terminal, then:"
echo "   transcript video.mp4"
echo "   transcript video.mp4 -f srt"
