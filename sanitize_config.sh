#!/bin/bash

# Script to sanitize configuration before pushing to GitHub
# Removes API keys and sensitive information

echo "Creating sanitized version of config files..."

# Check if config directory exists
if [ -d "config" ]; then
    # Check if the settings file contains API keys
    if grep -q "POLYGON_API_KEY" config/settings.yaml; then
        echo "Found API key in settings.yaml, creating sanitized version..."
        # Create a sanitized copy
        sed '/POLYGON_API_KEY/d' config/settings.yaml > config/settings_sanitized.yaml
        # Replace the original
        mv config/settings_sanitized.yaml config/settings.yaml
        echo "Sanitized settings.yaml"
    else
        echo "No API keys found in settings.yaml"
    fi
    
    # Look for any other config files with potential keys
    for file in config/*.yaml config/*.json; do
        if [ -f "$file" ] && grep -q -E "api_key|secret|token|password|credential" "$file"; then
            echo "Found sensitive data in $file, creating sanitized version..."
            sed -E 's/(api_key|secret|token|password|credential).*/"REMOVED_FOR_SECURITY"/g' "$file" > "${file}_sanitized"
            mv "${file}_sanitized" "$file"
            echo "Sanitized $file"
        fi
    done
fi

# Check for environment files
if [ -f ".env" ]; then
    echo "Found .env file, adding to .gitignore..."
    echo ".env" >> .gitignore
fi

# Check source files for hardcoded keys
echo "Checking source files for hardcoded API keys..."
for src_file in $(find src -name "*.py"); do
    if grep -q -E "api_key|secret|token|password|credential" "$src_file"; then
        echo "Found potential sensitive data in $src_file, creating sanitized version..."
        # Replace key values but keep variable names
        sed -E 's/(api_key|secret|token|password|credential).*= *"[^"]*"/"REMOVED_FOR_SECURITY"/g' "$src_file" > "${src_file}_sanitized"
        mv "${src_file}_sanitized" "$src_file"
        echo "Sanitized $src_file"
    fi
done

echo "Config sanitization complete!"