#!/bin/bash

# Script to delete all files ending with '_epo.fif'
# This script provides a safe way to remove MNE-Python epoch files
# with dry-run mode and user confirmation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dry-run    Show files that would be deleted without actually deleting them"
    echo "  -f, --force      Skip confirmation prompt and delete files immediately"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --dry-run     # Preview files that would be deleted"
    echo "  $0               # Interactive mode with confirmation"
    echo "  $0 --force       # Delete all _epo.fif files without confirmation"
    echo ""
    echo "This script will recursively search for and delete all files ending with '_epo.fif'"
    echo "in the current directory and all subdirectories."
}

# Function to find all _epo.fif files
find_epo_files() {
    local search_dir="${1:-.}"
    find "$search_dir" -type f -name "*_epo.fif" 2>/dev/null | sort
}

# Function to count files
count_files() {
    local files="$1"
    echo "$files" | grep -c "^" || echo "0"
}

# Function to get total size of files
get_total_size() {
    local files="$1"
    if [ -z "$files" ]; then
        echo "0"
        return
    fi
    
    local total_size=0
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
            total_size=$((total_size + size))
        fi
    done <<< "$files"
    echo "$total_size"
}

# Function to format file size
format_size() {
    local size="$1"
    if [ "$size" -gt 1073741824 ]; then
        echo "$(echo "scale=2; $size/1073741824" | bc) GB"
    elif [ "$size" -gt 1048576 ]; then
        echo "$(echo "scale=2; $size/1048576" | bc) MB"
    elif [ "$size" -gt 1024 ]; then
        echo "$(echo "scale=2; $size/1024" | bc) KB"
    else
        echo "${size} bytes"
    fi
}

# Function to delete files
delete_files() {
    local files="$1"
    local deleted_count=0
    local error_count=0
    
    print_info "Starting deletion process..."
    
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            if rm "$file" 2>/dev/null; then
                print_success "Deleted: $file"
                ((deleted_count++))
            else
                print_error "Failed to delete: $file"
                ((error_count++))
            fi
        fi
    done <<< "$files"
    
    echo ""
    print_info "Deletion completed:"
    print_success "Successfully deleted: $deleted_count files"
    if [ "$error_count" -gt 0 ]; then
        print_error "Failed to delete: $error_count files"
    fi
}

# Parse command line arguments
DRY_RUN=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main script logic
echo "=========================================="
echo "    _epo.fif File Deletion Script"
echo "=========================================="
echo ""

# Find all _epo.fif files
print_info "Searching for files ending with '_epo.fif'..."
epo_files=$(find_epo_files ".")
file_count=$(count_files "$epo_files")

if [ "$file_count" -eq 0 ]; then
    print_info "No files ending with '_epo.fif' found."
    exit 0
fi

# Calculate total size
total_size=$(get_total_size "$epo_files")
formatted_size=$(format_size "$total_size")

print_info "Found $file_count files ending with '_epo.fif'"
print_info "Total size: $formatted_size"
echo ""

# Show files that would be deleted
echo "Files to be deleted:"
echo "==================="
echo "$epo_files" | sed 's/^/  /'
echo ""

# Dry run mode
if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - No files will be deleted"
    print_info "Files that would be deleted: $file_count"
    print_info "Total size that would be freed: $formatted_size"
    exit 0
fi

# Force mode or interactive confirmation
if [ "$FORCE" = false ]; then
    echo -n "Do you want to proceed with deletion? (y/N): "
    read -r response
    
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_info "Deletion cancelled by user."
        exit 0
    fi
    echo ""
fi

# Perform deletion
delete_files "$epo_files"

print_success "Script completed successfully!"
