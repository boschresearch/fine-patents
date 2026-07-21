#!/bin/bash

# Download all USPTO bulk data files listed in a JSON file using the USPTO ODP API key
# Before running this, get bulk metadata as described here https://data.uspto.gov/apis/bulk-data/search


set -e  # Exit on error

# Configuration
JSON_FILE="${1:-../../data/uspto/APPXML.json}"
JSON_BASENAME=$(basename "$JSON_FILE" .json)
DOWNLOAD_DIR="${2:-../../data/uspto/$JSON_BASENAME}"
MAX_CONCURRENT="${3:-3}"
API_KEY="${4:-${USPTO_API_KEY:-PLACEHOLDER}}"

# Check if API key is provided
if [ -z "$API_KEY" ]; then
    echo "Error: USPTO API key is required"
    echo "Please set the USPTO_API_KEY environment variable:"
    echo "  export USPTO_API_KEY='your-api-key-here'"
    echo ""
    echo "Or pass it as the 4th argument:"
    echo "  $0 [json_file] [download_dir] [max_concurrent] [api_key]"
    exit 1
fi

# Check if JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file not found: $JSON_FILE"
    echo "Usage: $0 [json_file] [download_dir] [max_concurrent] [api_key]"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install jq to parse JSON."
    echo "  On Ubuntu/Debian: sudo apt-get install jq"
    echo "  On macOS: brew install jq"
    exit 1
fi

# Create download directory
mkdir -p "$DOWNLOAD_DIR"

echo "Reading download URLs from: $JSON_FILE"
echo "Download directory: $DOWNLOAD_DIR"
echo "Max concurrent downloads: $MAX_CONCURRENT"
echo ""

# Extract all download URLs and filenames from JSON
mapfile -t URLS < <(jq -r '.bulkDataProductBag[0].productFileBag.fileDataBag[].fileDownloadURI' "$JSON_FILE")
mapfile -t FILENAMES < <(jq -r '.bulkDataProductBag[0].productFileBag.fileDataBag[].fileName' "$JSON_FILE")
mapfile -t FILESIZES < <(jq -r '.bulkDataProductBag[0].productFileBag.fileDataBag[].fileSize' "$JSON_FILE")

TOTAL_FILES=${#URLS[@]}
echo "Found $TOTAL_FILES files to download"
echo ""

# Function to format file size
format_size() {
    local size=$1
    if [ "$size" -lt 1024 ]; then
        echo "${size}B"
    elif [ "$size" -lt 1048576 ]; then
        echo "$(( size / 1024 ))KB"
    elif [ "$size" -lt 1073741824 ]; then
        echo "$(( size / 1048576 ))MB"
    else
        echo "$(( size / 1073741824 ))GB"
    fi
}

# Function to download a single file
download_file() {
    local url=$1
    local filename=$2
    local filesize=$3
    local index=$4

    local filepath="$DOWNLOAD_DIR/$filename"

    # Skip if file already exists and has correct size
    if [ -f "$filepath" ]; then
        local existing_size
        existing_size=$(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null)
        if [ "$existing_size" = "$filesize" ]; then
            echo "[$index/$TOTAL_FILES] Skipping $filename (already downloaded)"
            return 0
        else
            echo "[$index/$TOTAL_FILES] Re-downloading $filename (size mismatch: $existing_size vs $filesize)"
        fi
    else
        echo "[$index/$TOTAL_FILES] Downloading $filename ($(format_size "$filesize"))"
    fi

    # Download with curl, showing progress and passing API key
    if curl -L --fail --progress-bar \
        -H "X-API-KEY: $API_KEY" \
        -o "$filepath.tmp" "$url"; then
        mv "$filepath.tmp" "$filepath"
        echo "[$index/$TOTAL_FILES] Completed: $filename"
    else
        echo "[$index/$TOTAL_FILES] Failed: $filename"
        rm -f "$filepath.tmp"
        return 1
    fi
}

# Export functions and variables for parallel execution
export -f download_file format_size
export DOWNLOAD_DIR TOTAL_FILES API_KEY

# Download files with parallel execution
COUNTER=0
PIDS=()

for i in "${!URLS[@]}"; do
    COUNTER=$((i + 1))

    # Wait if we've reached max concurrent downloads
    while [ ${#PIDS[@]} -ge "$MAX_CONCURRENT" ]; do
        for pid_idx in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$pid_idx]}" 2>/dev/null; then
                unset 'PIDS[$pid_idx]'
            fi
        done
        PIDS=("${PIDS[@]}")  # Re-index array
        sleep 0.1
    done

    # Start download in background
    download_file "${URLS[$i]}" "${FILENAMES[$i]}" "${FILESIZES[$i]}" "$COUNTER" &
    PIDS+=($!)
done

# Wait for all background jobs to complete
echo ""
echo "Waiting for remaining downloads to complete..."
wait

echo ""
echo "Download complete! Files saved to: $DOWNLOAD_DIR"
echo "Total files: $TOTAL_FILES"


