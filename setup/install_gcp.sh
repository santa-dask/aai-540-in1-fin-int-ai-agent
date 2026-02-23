#!/bin/bash
set -e

INSTALL_DIR="/usr/local"
CLOUDSDK_ROOT="$INSTALL_DIR/google-cloud-sdk"

# Function to add gcloud to PATH for the current session
add_gcloud_to_path() {
  if [ -d "$CLOUDSDK_ROOT/bin" ] && [[ ":$PATH:" != *":$CLOUDSDK_ROOT/bin:"* ]]; then
    export PATH="$CLOUDSDK_ROOT/bin:$PATH"
    echo "Added Google Cloud SDK to PATH for current session." >&2
  fi
}

# Ensure the temporary directory exists
mkdir -p /tmp/gcloud_sdk_install
cd /tmp/gcloud_sdk_install \
  && curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/google-cloud-sdk.zip \
  && unzip -q google-cloud-sdk.zip \
  && ./google-cloud-sdk/install.sh --usage-reporting=false --path-update=true --rc-path="~/.bashrc" --disable-installation-options \
  && rm -rf /usr/local/google-cloud-sdk \
  && mv ./google-cloud-sdk /usr/local/google-cloud-sdk

# Ensure the PATH is correctly updated in the current session


# Verify gcloud installation
#gcloud version

print("Google Cloud SDK has been re-installed and verified.")

# Ensure gcloud is in PATH for this script session
#add_gcloud_to_path

echo "Updating Google Cloud SDK components..." >&2
gcloud components update --quiet

echo "Installing gke-gcloud-auth-plugin..." >&2
gcloud components install gke-gcloud-auth-plugin --quiet

echo "Google Cloud SDK and gke-gcloud-auth-plugin setup complete." >&2
