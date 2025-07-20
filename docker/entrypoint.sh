#!/bin/sh
set -e

# This script runs as root.

# Set ownership of the mounted cache volume to the appuser.
# This ensures the application has write permissions to its cache directories.
chown -R appuser:appuser /home/appuser/.cache

# Execute the main command (CMD) as the non-root 'appuser'.
exec su-exec appuser "$@"
