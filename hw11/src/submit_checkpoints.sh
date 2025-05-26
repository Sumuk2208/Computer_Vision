#!/usr/bin/env bash

# If -h or --help is passed, print the usage message
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <checkpoint_dir>"
    echo
    echo "Copies all checkpoints in the given directory to the submit directory. Must be run on one of the course servers."
    echo
    echo "Example:"
    echo "$ cd ~/csci631/assignments/assignment-11/src/  # or wherever the checkpoints are (maybe in /tmp/pycharm_project_XYZ/?)"
    echo "$ ./submit_checkpoints.sh ./logs/  # or whatever name you gave to the logs directory"
    exit 0
fi

SUBMIT_DIR="/home/grader/rlange-grd/submissions/"

# Assert that this is being run on one of the servers
SERVERS=("granger" "weasley" "lovegood")
ON_SERVER=false
for server in "${SERVERS[@]}"; do
    if [[ $(hostname) == *"$server"* ]]; then
        ON_SERVER=true
    fi
done

if [ "$ON_SERVER" = false ]; then
    echo "This script must be run on one of the servers."
    exit 1
fi

# Check that the user has provided a checkpoint directory
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <checkpoint_dir>"
    exit 1
fi

# Check that the checkpoint directory exists
if [ ! -d "$1" ]; then
    echo "Checkpoint directory $1 does not exist."
    exit 1
fi

# Generate a list of checkpoints to be copied
CHECKPOINTS=()
while IFS='' read -r line; do CHECKPOINTS+=("$line"); done < <(find "$1" -name "checkpoint_*.pt")

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "No checkpoints found in $1."
    exit 1
fi

echo "About to copy the following checkpoints to the submit directory:"
for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "  $checkpoint"
done

read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Copy the checkpoints to the submit directory
mkdir -p "$SUBMIT_DIR/$USER/checkpoints/"
for checkpoint in "${CHECKPOINTS[@]}"; do
    mkdir -p "$(dirname "$SUBMIT_DIR/$USER/checkpoints/${checkpoint/$1/}")"
    cp "$checkpoint" "$SUBMIT_DIR/$USER/checkpoints/${checkpoint/$1/}"
    # Echo '.' for each checkpoint copied
    echo -n '.'
done
echo # Newline

# Make them readable
chmod -R g+rwX "$SUBMIT_DIR/$USER/checkpoints/"

echo "Done. The following files have been submitted:"
tree "$SUBMIT_DIR/$USER/checkpoints/"
