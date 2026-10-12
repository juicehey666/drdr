#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${PROJECT_ROOT}/data/raw/mbrset_v1.0"
TARGET_DIR="${DATASET_ROOT}/archive"
SOURCE_URL="https://physionet.org/files/mbrset/1.0/"
PHYSIONET_USER="${PHYSIONET_USER:-bottle}"
OFFICIAL_CHECKSUMS="${TARGET_DIR}/SHA256SUMS.txt"

mkdir -p "${TARGET_DIR}"

echo "Downloading mBRSET v1.0 into: ${TARGET_DIR}"
echo "PhysioNet user: ${PHYSIONET_USER}"
echo "The password prompt is handled by wget and is not stored by this script."
echo "PhysioNet traffic uses a direct connection; local proxy settings are unchanged."

if [[ ! -s "${OFFICIAL_CHECKSUMS}" ]]; then
  echo "Official checksum file is missing: ${OFFICIAL_CHECKSUMS}" >&2
  echo "Run the full repository downloader once to obtain SHA256SUMS.txt." >&2
  exit 1
fi

URL_LIST="$(mktemp)"
VERIFY_LOG="$(mktemp)"
trap 'rm -f "${URL_LIST}" "${VERIFY_LOG}"' EXIT

echo "Checking existing files locally against the official SHA-256 list..."
(
  cd "${TARGET_DIR}"
  sha256sum -c SHA256SUMS.txt --ignore-missing --quiet
) >"${VERIFY_LOG}" 2>&1 || true

declare -A INVALID_FILES=()
while IFS= read -r failed_path; do
  [[ -n "${failed_path}" ]] && INVALID_FILES["${failed_path}"]=1
done < <(sed -n 's/: FAILED$//p' "${VERIFY_LOG}")

missing_count=0
invalid_count=0
while read -r _checksum relative_path; do
  relative_path="${relative_path#\*}"
  if [[ ! -f "${TARGET_DIR}/${relative_path}" ]]; then
    printf '%s%s\n' "${SOURCE_URL}" "${relative_path}" >>"${URL_LIST}"
    ((missing_count += 1))
  elif [[ -n "${INVALID_FILES[${relative_path}]+present}" ]]; then
    printf '%s%s\n' "${SOURCE_URL}" "${relative_path}" >>"${URL_LIST}"
    ((invalid_count += 1))
  fi
done <"${OFFICIAL_CHECKSUMS}"

download_count=$((missing_count + invalid_count))
echo "Missing files: ${missing_count}; incomplete/corrupt files: ${invalid_count}."

if (( download_count == 0 )); then
  echo "All files are already present and match the official checksums."
else
  echo "Downloading only ${download_count} missing or incomplete files..."

  env \
    -u http_proxy \
    -u https_proxy \
    -u HTTP_PROXY \
    -u HTTPS_PROXY \
    -u all_proxy \
    -u ALL_PROXY \
    wget \
    --input-file "${URL_LIST}" \
    --continue \
    --retry-connrefused \
    --waitretry=5 \
    --timeout=60 \
    --tries=0 \
    --force-directories \
    --no-host-directories \
    --cut-dirs=3 \
    --user "${PHYSIONET_USER}" \
    --ask-password \
    --directory-prefix "${TARGET_DIR}"
fi

echo "Verifying the complete dataset against the official SHA-256 list..."
(
  cd "${TARGET_DIR}"
  sha256sum -c SHA256SUMS.txt --quiet
)

echo "Generating a local SHA-256 manifest..."
find "${TARGET_DIR}" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${DATASET_ROOT}/checksums.sha256"

echo "mBRSET v1.0 download and checksum generation completed."
