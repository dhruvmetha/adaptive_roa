# retry N CMD...: run CMD up to N times, RETRY_SLEEP (default 5) seconds apart, and return
# its last exit status. The Amarel log rsync drops its ssh connection now and then
# (broken pipe, rsync 255/19/20), and a second try goes through.
retry() {
  local n=$1 i rc=0
  shift
  for ((i = 1; i <= n; i++)); do
    "$@" && return 0
    rc=$?
    [ "$i" -lt "$n" ] && sleep "${RETRY_SLEEP:-5}"
  done
  return "$rc"
}
