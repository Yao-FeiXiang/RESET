#!/usr/bin/env bash
set -euo pipefail

JAVA="${JAVA:-java}"

MVN="${MVN:-mvn}" 

MAIN_CLASS="${MAIN_CLASS:-Main}"

JAVA_OPTS_DEFAULT=(
  "-Dfile.encoding=UTF-8"
  "-Dsun.stdout.encoding=UTF-8"
  "-Dsun.stderr.encoding=UTF-8"
)

LIB_DIR="target/lib"

CLASSPATH="target/classes:${LIB_DIR}/*"


echo "[1/3] Building project (compile) ..."
$MVN -q -DskipTests package

echo "[2/3] Preparing runtime dependencies into ${LIB_DIR} ..."
$MVN -q -DskipTests dependency:copy-dependencies -DoutputDirectory="${LIB_DIR}"


echo "[3/3] Running: ${MAIN_CLASS}"
echo "      CP: ${CLASSPATH}"
echo


JAVA_OPTS=("${JAVA_OPTS_DEFAULT[@]}")
if [[ -n "${JVM_OPTS:-}" ]]; then
  JAVA_OPTS+=(${JVM_OPTS})
fi

exec $JAVA "${JAVA_OPTS[@]}" -cp "${CLASSPATH}" "${MAIN_CLASS}" "$@"
