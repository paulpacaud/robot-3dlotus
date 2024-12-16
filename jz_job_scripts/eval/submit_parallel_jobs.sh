#!/bin/bash

cd $WORK/Projects/robot-3dlotus

# Check if MODEL is provided
if [ -z "$MODEL" ]; then
    echo "Error: MODEL parameter is required"
    exit 1
fi

# Determine the evaluation script based on MODEL
case $MODEL in
    "3dlotus")
        EVAL_TASK_SCRIPT="jz_job_scripts/eval/simple_policy/eval_3dlotus_single_taskvar.sh"
        ;;
    "robot_pipeline")
        EVAL_TASK_SCRIPT="jz_job_scripts/eval/robot_pipeline/eval_robot_pipeline_single_taskvar.sh"
        ;;
    *)
        echo "Error: MODEL must be either '3dlotus' or 'robot_pipeline'"
        exit 1
        ;;
esac

LOG_DIR="slurm_monitoring"
# Create a unique filename using timestamp
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
JOBS_FILE="${LOG_DIR}/jobs_${TIMESTAMP}.txt"

# Create monitoring directory
mkdir -p "${LOG_DIR}"

# Initialize our jobs tracking file with headers
echo "JOBID,TASKVAR,SUBMIT_TIME,STATUS,PORT" > "${JOBS_FILE}"

# Create a temporary directory for job scripts
TEMP_DIR=$(mktemp -d)

# Initialize base port number
BASE_PORT=15324
PORT_COUNTER=0

# Submit jobs and track them
cat "${taskfile}" | while IFS=, read -r taskvar rest; do
    # Skip header and empty lines
    taskvar=$(echo "$taskvar" | tr -d '[:space:]\r')
    if [[ -z "$taskvar" ]] || [[ "$taskvar" == "taskvar" ]]; then
        continue
    fi

    # Calculate unique port for this job
    CURRENT_PORT=$((BASE_PORT + PORT_COUNTER))
    PORT_COUNTER=$((PORT_COUNTER + 1))

    # Create and modify job script
    job_script="${TEMP_DIR}/job_${taskvar//+/_}.sh"
    cp "$EVAL_TASK_SCRIPT" "$job_script"
    sed -i "s/taskvar=.*/taskvar=${taskvar}/" "$job_script"
    # Replace the port number in the script
    sed -i "s/llm_port=.*/llm_port=${CURRENT_PORT}/" "$job_script"

    # Submit job and capture the job ID
    job_id=$(sbatch --parsable "$job_script")
    submit_time=$(date '+%Y-%m-%d_%H:%M:%S')

    # Record job information including the port number
    echo "${job_id},${taskvar},${submit_time},PENDING,${CURRENT_PORT}" >> "${JOBS_FILE}"

    echo "Submitted job ${job_id} for task: ${taskvar} with port: ${CURRENT_PORT}"
done

# echo the jobs file name for reference
echo "Job tracking file: ${JOBS_FILE}"
echo "Use the following command to monitor job status:"
echo "chmod +x ./jz_job_scripts/eval/check_jobs.sh; ./jz_job_scripts/eval/check_jobs.sh ${JOBS_FILE}"

# Clean up temporary directory
rm -rf "$TEMP_DIR"