#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/root/client_data_snapshots}"
PYTHON_BIN="${PYTHON_BIN:-/root/oracle-env/bin/python}"
EXPORT_SCRIPT="${EXPORT_SCRIPT:-/root/export_oracle_snapshot.py}"
INSPECT_SCRIPT="${INSPECT_SCRIPT:-/root/inspect_oracle_tables.py}"
LEGACY_CONFIG="${LEGACY_CONFIG:-/root/get_store.py}"
DATE_TAG="${1:-$(date +%Y%m%d)}"

OUT_DIR="${SNAPSHOT_ROOT}/client_snapshot_${DATE_TAG}"
LOG_DIR="${SNAPSHOT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/daily_oracle_snapshot_${DATE_TAG}.log"
LOCK_FILE="/tmp/b2b_oracle_snapshot.lock"
SUCCESS_FILE="${OUT_DIR}/_SUCCESS"
ARCHIVE_PATH="${SNAPSHOT_ROOT}/client_snapshot_${DATE_TAG}.tar.gz"
ARCHIVE_SHA_PATH="${ARCHIVE_PATH}.sha256"
INVENTORY_PATH="${SNAPSHOT_ROOT}/oracle_table_inventory_${DATE_TAG}.json"

mkdir -p "${SNAPSHOT_ROOT}" "${LOG_DIR}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[$(date -Is)] another snapshot job is running; exit"
  exit 0
fi

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date -Is)] start daily Oracle snapshot: ${DATE_TAG}"
echo "SNAPSHOT_ROOT=${SNAPSHOT_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[$(date -Is)] DRY_RUN=1, validation only"
  test -x "${PYTHON_BIN}"
  test -f "${EXPORT_SCRIPT}"
  test -f "${INSPECT_SCRIPT}"
  test -f "${LEGACY_CONFIG}"
  echo "[$(date -Is)] dry run passed"
  exit 0
fi

if [[ -f "${SUCCESS_FILE}" && "${FORCE:-0}" != "1" ]]; then
  echo "[$(date -Is)] snapshot already marked successful; skip: ${SUCCESS_FILE}"
  exit 0
fi

"${PYTHON_BIN}" "${INSPECT_SCRIPT}" "${LEGACY_CONFIG}" "${INVENTORY_PATH}"

"${PYTHON_BIN}" "${EXPORT_SCRIPT}" \
  --output-dir "${OUT_DIR}" \
  --legacy-config "${LEGACY_CONFIG}"

cp "${INVENTORY_PATH}" "${OUT_DIR}/oracle_table_inventory_${DATE_TAG}.json"

tar -czf "${ARCHIVE_PATH}" -C "${SNAPSHOT_ROOT}" "client_snapshot_${DATE_TAG}"
sha256sum "${ARCHIVE_PATH}" > "${ARCHIVE_SHA_PATH}"

{
  echo "completed_at=$(date -Is)"
  echo "snapshot_dir=${OUT_DIR}"
  echo "archive=${ARCHIVE_PATH}"
  echo "archive_sha256=$(cut -d ' ' -f 1 "${ARCHIVE_SHA_PATH}")"
} > "${SUCCESS_FILE}"

echo "[$(date -Is)] completed daily Oracle snapshot: ${DATE_TAG}"
echo "archive=${ARCHIVE_PATH}"
echo "sha256=$(cut -d ' ' -f 1 "${ARCHIVE_SHA_PATH}")"
