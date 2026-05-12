#!/bin/sh
set -e

APIS_DIR="/apis"
MICROCKS_URL="${MICROCKS_URL:-http://mock-server:8080/api}"
API_IMPORTER_URL="${API_IMPORTER_URL:-http://api-importer:7500}"
TOKEN="${TOKEN:-dummy}"

echo "| Starting import of APIs into Microcks..."

for api_file in "$APIS_DIR"/*.yaml; do
  api_filename=${api_file##*/}
  base_name=${api_filename%.yaml}

  raw_title=$(grep -m 1 "^[[:space:]]*title:" "$api_file" \
    | sed 's/^[[:space:]]*title:[[:space:]]*//' \
    | tr -d "'\"" | tr -d '\r')

  echo "| Importing: $api_filename ($raw_title)"
  echo ""

  # Import into Microcks
  microcks import "${api_file}:true" \
    --microcksURL="${MICROCKS_URL}" \
    --keycloakClientId=foo --keycloakClientSecret=bar \
    2>/dev/null
  echo "| Imported into Microcks."

  # Register via api-importer
  echo "| Registering in catalog via api-importer..."
  curl -s -X POST "${API_IMPORTER_URL}/api/importer/import/file" \
    -F "file=@${api_file}" \
    -F "id=${base_name}" \
    -F "base_url=http://mock-server:8080/rest" \
    2>/dev/null || echo "| Warning: api-importer registration failed, continuing..."

  echo ""
  echo "-----------------------------------"
done

echo "| Import complete."
