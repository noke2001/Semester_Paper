#!/bin/bash

# example usage: 
# bash download.sh

# --- BASE CONFIG ---
export BASE_CACHE="./openml_cache"
export PROJECT_ROOT="./main"
export CONTAINER="SMAC_optuna.sif"

# --- TASK/SUITE ---
export TASK_ID="361111"
export SUITE_ID="334"

echo "--- FETCHING METADATA ---"
METADATA=$(apptainer exec --cleanenv \
	--env PYTHONNOUSERSITE=1 \
	--env XDG_CACHE_HOME=$BASE_CACHE \
	$CONTAINER \
	python3 -c "import openml; \
	t=openml.tasks.get_task($TASK_ID, download_data=False); \
	d=openml.datasets.get_dataset(t.dataset_id, download_data=False); \
	print(f'{t.dataset_id} {d.url}')")

# --- Fail Check ---
if [ $? -ne 0 ]; then
	echo "ERROR fetching metadeta. Check Task ID."
	exit 1
fi

# --- Parse Output ---
read -r DS_ID DS_URL <<< "$METADATA"
echo "	-> Dataset ID: $DS_ID"
echo "	-> URL: $DS_URL"

echo "--- DOWNLOADING RAW DATA ---"
TARGET_DIR="$BASE_CACHE/openml/org/openml/www/datasets/$DS_ID"
TARGET_FILE="$TARGET_DIR/dataset.arff"
rm -f "$TARGET_FILE"
mkdir -p "$TARGET_DIR"
echo "	-> Downloading to $TARGET_FILE ..."

curl -L -f -o "$TARGET_FILE" "$DS_URL"

if [ $? -ne 0 ]; then
	echo "ERROR download failed (Network or URL error)"
	exit 1
fi

echo "--- VERIFYING FILE INTEGRITY ---"
FIRST_LINE=$(head -n 1 "$TARGET_FILE")
echo "	-> First line of file: $FIRST_LINE"

# Check for specifica OpenML Database Error
if [[ "$FIRST_LINE" == *"Database connection error"* ]]; then
	echo "SERVER ERROR: OpenML is overloaded"
	echo "	Action: Wait 1 minute and try again."
	exit 1
fi

# Check for HTML/JSON
if [[ "$FIRST_LINE" == *"<"* ]] || [[ "$FIRST_LINE" == *"{"* ]]; then
	echo "CRITICAL ERROR: The file is corrupted (looks like HTML or JSON)."
	echo "	This usually means the OpenML URL redirected to an error page."
	exit 1
fi

# Strict Check
LOWER_LINE=$(echo "$FIRST_LINE" | tr '[:upper:]' '[:lower:]')
if [["$LOWER_LINE" != %* ]] && [[ "$LOWER_LINE" != @relation* ]]; then
	echo "CRITICAL ERROR: File does not look like an ARFF file."
	echo 1
fi

echo "File looks Valid"

echo "--- PROCESSING DATA (ARFF -> CSV) ---"
apptainer exec --cleanenv \
	--env PYTHONNOUSERSITE=1 \
	--env NUMBA_CACHE_DIR=/tmp/numba_cache \
	--env XDG_CACHE_HOME=$BASE_CACHE \
	$CONTAINER \
	python3 $PROJECT_ROOT/download_data.py --task_id $TASK_ID --suite_id $SUITE_ID

echo "--- READY FOR EXPERIMENT ---"
