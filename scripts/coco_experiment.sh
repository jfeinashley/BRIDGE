#!/bin/bash

# COCO Retrieval Experiment Script
# Runs training and evaluation in a tmux session with full logging

# Configuration Variables
MODEL_SIZE=${1:-base}              # Model size: base, large, or huge
PRETRAINED=${2:-""}                # Path to pretrained checkpoint (optional)
GPUS=${3:-"0,1,2,3"}                     # GPUs to use (e.g., "0,1,2,3" or "0")
SAVE_FREQ=${4:-5}                  # Save checkpoint every N epochs
EXPERIMENT_NAME=${5:-"exp_$(date +%Y%m%d_%H%M%S)"}  # Experiment name

# Set paths based on model size
if [ "$MODEL_SIZE" = "base" ]; then
    CONFIG="configs/coco_base.yaml"
    # CONFIG="configs/coco_exp1.yaml"
    OUTPUT_BASE="output/COCO_Retrieval_Base"
elif [ "$MODEL_SIZE" = "large" ]; then
    CONFIG="configs/coco_large.yaml"
    OUTPUT_BASE="output/COCO_Retrieval_Large"
elif [ "$MODEL_SIZE" = "huge" ]; then
    CONFIG="configs/coco_huge.yaml"
    OUTPUT_BASE="output/COCO_Retrieval_Huge"
else
    echo "Error: Invalid model size. Use: base, large, or huge"
    echo "Usage: $0 [base|large|huge] [pretrained_path] [gpus] [save_freq] [experiment_name]"
    echo "Example: $0 base pretrained.pth 0,1,2,3 5 my_experiment"
    exit 1
fi

# Create experiment directory
OUTPUT_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}"
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/logs

# Set log files
LOG_FILE="${OUTPUT_DIR}/logs/training_$(date +%Y%m%d_%H%M%S).log"
ERROR_LOG="${OUTPUT_DIR}/logs/error_$(date +%Y%m%d_%H%M%S).log"
VAL_STATS="${OUTPUT_DIR}/logs/validation_stats.txt"
PARAM_LOG="${OUTPUT_DIR}/logs/parameter_analysis.txt"

# Set GPU environment
export CUDA_VISIBLE_DEVICES=${GPUS}

# Count number of GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Tmux session name
SESSION_NAME="coco_retrieval_${MODEL_SIZE}_${EXPERIMENT_NAME}"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install it first:"
    echo "  Ubuntu/Debian: sudo apt-get install tmux"
    echo "  CentOS/RHEL: sudo yum install tmux"
    echo "  macOS: brew install tmux"
    exit 1
fi

# Check if session already exists
if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo "Error: Tmux session '${SESSION_NAME}' already exists"
    echo "Please attach to it with: tmux attach -t ${SESSION_NAME}"
    echo "Or kill it with: tmux kill-session -t ${SESSION_NAME}"
    exit 1
fi

# Print experiment configuration
{
    echo "========================================"
    echo "COCO RETRIEVAL EXPERIMENT"
    echo "========================================"
    echo "Experiment: ${EXPERIMENT_NAME}"
    echo "Model Size: ${MODEL_SIZE}"
    echo "Config: ${CONFIG}"
    echo "Output Dir: ${OUTPUT_DIR}"
    echo "Pretrained: ${PRETRAINED:-None}"
    echo "GPUs: ${GPUS} (${NUM_GPUS} GPUs)"
    echo "Save Frequency: Every ${SAVE_FREQ} epochs"
    echo "Tmux Session: ${SESSION_NAME}"
    echo "Start Time: $(date)"
    echo "========================================"
    echo ""
} | tee ${LOG_FILE}

# Build the command based on number of GPUs and pretrained model
if [ ${NUM_GPUS} -gt 1 ]; then
    # Multi-GPU training with distributed using torchrun
    if [ -z "$PRETRAINED" ]; then
        # No pretrained model - start from scratch
        CMD="torchrun \
            --nproc_per_node=${NUM_GPUS} \
            --master_port=29500 \
            train/experiment_retrieval.py \
            --config ${CONFIG} \
            --output_dir ${OUTPUT_DIR} \
            --save_freq ${SAVE_FREQ} \
            --distributed True"
    else
        # With pretrained model
        CMD="torchrun \
            --nproc_per_node=${NUM_GPUS} \
            --master_port=29500 \
            train/experiment_retrieval.py \
            --config ${CONFIG} \
            --output_dir ${OUTPUT_DIR} \
            --pretrained ${PRETRAINED} \
            --save_freq ${SAVE_FREQ} \
            --distributed True"
    fi
else
    # Single GPU training
    if [ -z "$PRETRAINED" ]; then
        # No pretrained model - start from scratch
        CMD="python train/experiment_retrieval.py \
            --config ${CONFIG} \
            --output_dir ${OUTPUT_DIR} \
            --save_freq ${SAVE_FREQ} \
            --distributed False"
    else
        # With pretrained model
        CMD="python train/experiment_retrieval.py \
            --config ${CONFIG} \
            --output_dir ${OUTPUT_DIR} \
            --pretrained ${PRETRAINED} \
            --save_freq ${SAVE_FREQ} \
            --distributed False"
    fi
fi

# Save session info
echo ${SESSION_NAME} > ${OUTPUT_DIR}/session_name.txt
echo ${LOG_FILE} > ${OUTPUT_DIR}/log_path.txt

# Create tmux session and run training
echo "Creating tmux session: ${SESSION_NAME}"
echo "Command: ${CMD}"
echo ""

# Create the tmux session in detached mode
tmux new-session -d -s ${SESSION_NAME}

# Set GPU environment in tmux session
tmux send-keys -t ${SESSION_NAME} "export CUDA_VISIBLE_DEVICES=${GPUS}" C-m

# Send the training command with logging
tmux send-keys -t ${SESSION_NAME} "${CMD} 2>&1 | tee ${LOG_FILE}" C-m

# Create attach script
cat > ${OUTPUT_DIR}/attach.sh << EOF
#!/bin/bash
echo "Attaching to tmux session: ${SESSION_NAME}"
echo "To detach: Press Ctrl+B, then D"
echo ""
tmux attach -t ${SESSION_NAME}
EOF
chmod +x ${OUTPUT_DIR}/attach.sh

# Create a status script
cat > ${OUTPUT_DIR}/status.sh << 'EOFSTATUS'
#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SESSION_NAME=$(cat ${SCRIPT_DIR}/session_name.txt 2>/dev/null)
LOG_FILE=$(cat ${SCRIPT_DIR}/log_path.txt 2>/dev/null)
VAL_STATS="${SCRIPT_DIR}/logs/validation_stats.txt"

if [ -z "${SESSION_NAME}" ]; then
    echo "Error: Session name not found"
    exit 1
fi

# Check if tmux session is running
if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo "Training is RUNNING in tmux session: ${SESSION_NAME}"
    echo "To attach: tmux attach -t ${SESSION_NAME}"
    echo "Or use: ${SCRIPT_DIR}/attach.sh"
else
    echo "Tmux session '${SESSION_NAME}' has ENDED"
    echo "Training may be complete or stopped"
fi

if [ -f "${LOG_FILE}" ]; then
    echo ""
    echo "Latest training status:"
    echo "----------------------"
    tail -n 30 ${LOG_FILE} | grep -E "Epoch|loss|Mean: R@|best|COMPLETE"
    
    # Extract validation stats
    grep -E "EPOCH.*RESULTS:|Mean: R@" ${LOG_FILE} | tail -n 20 > ${VAL_STATS} 2>/dev/null
    
    if [ -f "${VAL_STATS}" ]; then
        echo ""
        echo "Validation performance summary:"
        echo "------------------------------"
        tail -n 10 ${VAL_STATS}
    fi
    
    # Show training progress
    echo ""
    echo "Training progress:"
    echo "------------------"
    grep "Epoch [0-9]" ${LOG_FILE} | tail -5
fi
EOFSTATUS
chmod +x ${OUTPUT_DIR}/status.sh

# Create a stop script
cat > ${OUTPUT_DIR}/stop.sh << EOFSTOP
#!/bin/bash
SCRIPT_DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )"
SESSION_NAME=\$(cat \${SCRIPT_DIR}/session_name.txt 2>/dev/null)

if [ -z "\${SESSION_NAME}" ]; then
    echo "Error: Session name not found"
    exit 1
fi

if tmux has-session -t \${SESSION_NAME} 2>/dev/null; then
    echo "Stopping tmux session: \${SESSION_NAME}"
    # Send Ctrl+C to the session
    tmux send-keys -t \${SESSION_NAME} C-c
    sleep 2
    # Kill the session
    tmux kill-session -t \${SESSION_NAME}
    echo "Training stopped."
else
    echo "Tmux session '\${SESSION_NAME}' is not running."
fi
EOFSTOP
chmod +x ${OUTPUT_DIR}/stop.sh

# Create a tail logs script
cat > ${OUTPUT_DIR}/tail_logs.sh << 'EOFTAIL'
#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_FILE=$(cat ${SCRIPT_DIR}/log_path.txt 2>/dev/null)

if [ -f "${LOG_FILE}" ]; then
    echo "Tailing log file: ${LOG_FILE}"
    echo "Press Ctrl+C to stop"
    echo ""
    tail -f ${LOG_FILE}
else
    echo "No log file found"
fi
EOFTAIL
chmod +x ${OUTPUT_DIR}/tail_logs.sh

# Create monitoring script that extracts stats periodically
cat > ${OUTPUT_DIR}/monitor.sh << 'EOFMON'
#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SESSION_NAME=$(cat ${SCRIPT_DIR}/session_name.txt 2>/dev/null)
LOG_FILE=$(cat ${SCRIPT_DIR}/log_path.txt 2>/dev/null)
VAL_STATS="${SCRIPT_DIR}/logs/validation_stats.txt"
PARAM_LOG="${SCRIPT_DIR}/logs/parameter_analysis.txt"

echo "Starting monitor for tmux session: ${SESSION_NAME}"
echo "Monitoring log: ${LOG_FILE}"
echo ""

while true; do
    if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
        # Extract validation stats
        grep -E "EPOCH.*RESULTS:|Mean: R@|Image→Text:|Text→Image:" ${LOG_FILE} > ${VAL_STATS} 2>/dev/null
        
        # Extract parameter analysis (only once, at the beginning)
        if [ ! -f "${PARAM_LOG}" ]; then
            sed -n '/MODEL PARAMETER ANALYSIS/,/TRAINING START/p' ${LOG_FILE} > ${PARAM_LOG} 2>/dev/null
        fi
        
        # Sleep for 5 minutes before next check
        sleep 300
    else
        echo "$(date): Training session ended"
        # Final extraction
        grep -E "EPOCH.*RESULTS:|Mean: R@|Image→Text:|Text→Image:" ${LOG_FILE} > ${VAL_STATS} 2>/dev/null
        break
    fi
done
EOFMON
chmod +x ${OUTPUT_DIR}/monitor.sh

# Start monitoring in background tmux session
MONITOR_SESSION="${SESSION_NAME}_monitor"
tmux new-session -d -s ${MONITOR_SESSION}
tmux send-keys -t ${MONITOR_SESSION} "cd $(pwd) && ${OUTPUT_DIR}/monitor.sh" C-m

# Final message
echo "========================================"
echo "EXPERIMENT LAUNCHED IN TMUX"
echo "========================================"
echo "Experiment directory: ${OUTPUT_DIR}"
echo "Tmux session: ${SESSION_NAME}"
echo ""
echo "Useful commands:"
echo "  Attach to session: tmux attach -t ${SESSION_NAME}"
echo "                     OR ${OUTPUT_DIR}/attach.sh"
echo "  Detach from session: Press Ctrl+B, then D"
echo "  Check status:  ${OUTPUT_DIR}/status.sh"
echo "  Tail logs:     ${OUTPUT_DIR}/tail_logs.sh" 
echo "  Stop training: ${OUTPUT_DIR}/stop.sh"
echo "  View logs:     less ${LOG_FILE}"
echo ""
echo "Tmux tips:"
echo "  List sessions:     tmux ls"
echo "  Kill session:      tmux kill-session -t ${SESSION_NAME}"
echo "  Scroll in tmux:    Ctrl+B then ["
echo ""
echo "The training is running in tmux session '${SESSION_NAME}'"
echo "You can safely close this terminal."
echo "========================================"