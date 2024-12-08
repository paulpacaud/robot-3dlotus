#!/bin/bash

JOBS_FILE="$1"
if [ -z "$JOBS_FILE" ]; then
    echo "Error: No jobs file specified"
    echo "Usage: $0 <jobs_file>"
    exit 1
fi
LOG_DIR="slurm_monitoring"
SUMMARY_FILE="${LOG_DIR}/job_summary.txt"

# Define color codes for better readability
GREEN='\033[0;32m'    # For completed jobs
YELLOW='\033[1;33m'   # For running jobs
RED='\033[0;31m'      # For failed jobs
BLUE='\033[0;94m'    # For headers and separators
GRAY='\033[0;90m'     # For log output
NC='\033[0m'          # No Color - used to reset coloring

calculate_duration() {
    local start_time="$1"
    local end_time="$2"

    # Return "Not started" if no start time
    if [ "$start_time" = "None" ] || [ "$start_time" = "Unknown" ]; then
        echo "Not started"
        return
    fi

    # Calculate duration based on whether job has finished
    if [ "$end_time" != "None" ] && [ "$end_time" != "Unknown" ]; then
        # For finished jobs
        local duration=$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))
    else
        # For running jobs
        local duration=$(( $(date +%s) - $(date -d "$start_time" +%s) ))
    fi

    # Format duration as HH:MM:SS
    printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60))
}

# Create a colorful header
echo -e "${BLUE}Job Status Summary ($(date '+%Y-%m-%d %H:%M:%S'))${NC}" > "${SUMMARY_FILE}"
echo -e "${BLUE}----------------------------------------${NC}" >> "${SUMMARY_FILE}"

# Process each job
tail -n +2 "${JOBS_FILE}" | while IFS=, read -r jobid taskvar submit_time status; do
    # Handle rejected jobs (no job ID)
    if [[ -z "$jobid" ]] || [[ ! "$jobid" =~ ^[0-9]+$ ]]; then
        echo -e "${BLUE}Task: ${NC}${taskvar}" >> "${SUMMARY_FILE}"
        echo -e "  ${BLUE}Job ID:${NC} N/A" >> "${SUMMARY_FILE}"
        echo -e "  ${BLUE}Status:${NC} ${RED}REJECTED${NC}" >> "${SUMMARY_FILE}"
        echo -e "  ${BLUE}Submit Time:${NC} ${submit_time}" >> "${SUMMARY_FILE}"
        echo -e "  ${BLUE}Duration:${NC} N/A" >> "${SUMMARY_FILE}"
        echo -e "${BLUE}----------------------------------------${NC}" >> "${SUMMARY_FILE}"
        continue
    fi

    job_info=$(sacct -j "${jobid}" --format=State,Start,End --noheader | head -n1)

    job_status=$(echo "$job_info" | awk '{print $1}' | tr -d '[:space:]')
    start_time=$(echo "$job_info" | awk '{print $2}' | tr -d '[:space:]')
    end_time=$(echo "$job_info" | awk '{print $3}' | tr -d '[:space:]')

    # Calculate duration
    if [ "$start_time" != "None" ] && [ "$start_time" != "Unknown" ]; then
        if [ "$end_time" != "None" ] && [ "$end_time" != "Unknown" ]; then
            duration=$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))
        else
            duration=$(( $(date +%s) - $(date -d "$start_time" +%s) ))
        fi
        duration_formatted=$(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60)))
    else
        duration_formatted="Not started"
    fi

    # Update status in jobs file
    sed -i "s/^${jobid},${taskvar},${submit_time},.*/${jobid},${taskvar},${submit_time},${job_status}/" "${JOBS_FILE}"

    # Choose color based on job status
    status_color="${YELLOW}"  # Default color for unknown states
    case "${job_status}" in
        "COMPLETED") status_color="${GREEN}" ;;
        "RUNNING")   status_color="${YELLOW}" ;;
        "FAILED"|"TIMEOUT"|"CANCELLED"|"CANCELLED+") status_color="${RED}" ;;
    esac

    # Add colored information to summary
    echo -e "${BLUE}Task: ${NC}${taskvar}" >> "${SUMMARY_FILE}"
    echo -e "  ${BLUE}Job ID:${NC} ${jobid}" >> "${SUMMARY_FILE}"
    echo -e "  ${BLUE}Status:${NC} ${status_color}${job_status}${NC}" >> "${SUMMARY_FILE}"
    echo -e "  ${BLUE}Submit Time:${NC} ${submit_time}" >> "${SUMMARY_FILE}"
    echo -e "  ${BLUE}Duration:${NC} ${duration_formatted}" >> "${SUMMARY_FILE}"

    # Add log file information and last 5 lines
    log_file="slurm_logs/${jobid}.out"
    if [ -f "${log_file}" ]; then
        echo -e "  ${BLUE}Log File:${NC} ${log_file}" >> "${SUMMARY_FILE}"
        echo -e "  ${BLUE}Last log lines:${NC}" >> "${SUMMARY_FILE}"
        echo -e "${GRAY}  $(tail -n 5 "${log_file}" | sed 's/^/    /')${NC}" >> "${SUMMARY_FILE}"
    fi

    echo -e "${BLUE}----------------------------------------${NC}" >> "${SUMMARY_FILE}"
done

# Print summary of all jobs with colors
echo -e "\n${BLUE}All Jobs Summary:${NC}" >> "${SUMMARY_FILE}"
cat "${JOBS_FILE}" | tail -n +2 | while IFS=, read -r jobid taskvar submit_time status; do
    # Handle rejected jobs in summary
    if [[ -z "$jobid" ]] || [[ ! "$jobid" =~ ^[0-9]+$ ]]; then
        echo -e "  N/A ${taskvar} N/A ${RED}REJECTED${NC}" >> "${SUMMARY_FILE}"
        continue
    fi

    # Get timing information for this job
    job_info=$(sacct -j "${jobid}" --format=State,Start,End --noheader | head -n1)
    start_time=$(echo "$job_info" | awk '{print $2}' | tr -d '[:space:]')
    end_time=$(echo "$job_info" | awk '{print $3}' | tr -d '[:space:]')

    # Calculate duration using our function
    duration_formatted=$(calculate_duration "$start_time" "$end_time")

    # Choose status color
    status_color="${YELLOW}"  # Default color
    case "${status}" in
        "COMPLETED") status_color="${GREEN}" ;;
        "RUNNING")   status_color="${YELLOW}" ;;
        "FAILED"|"TIMEOUT"|"CANCELLED"|"CANCELLED+") status_color="${RED}" ;;
    esac

    # Format each line with colored status and duration
    echo -e "  ${jobid} ${taskvar} ${duration_formatted} ${status_color}${status}${NC}" >> "${SUMMARY_FILE}"
done

# Print Job IDs list (excluding rejected jobs)
echo -n "Job IDs: " >> "${SUMMARY_FILE}"
tail -n +2 "${JOBS_FILE}" | cut -d',' -f1 | grep -E '^[0-9]+$' | tr '\n' ' ' >> "${SUMMARY_FILE}"
echo >> "${SUMMARY_FILE}"

cat "${SUMMARY_FILE}"