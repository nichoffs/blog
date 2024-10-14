#!/bin/bash

# Usage: ./convert_ipynb_to_html.sh <path_to_ipynb_file>

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_ipynb_file>"
    exit 1
fi

# Get the path to the IPYNB file
IPYNB_PATH=$1

# Convert the IPYNB file to HTML
jupyter nbconvert --to html "$IPYNB_PATH"

# Notify the user
echo "Conversion complete. HTML file is located in the same directory as the IPYNB file."
