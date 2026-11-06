"""Exact R candidate boundary for the Duan thesis task.

The official test calls ``source('/results/Duan_models.R')`` and then invokes
two R functions directly. This adapter makes source define thin R proxies;
the candidate source and function calls execute in a persistent bwrap worker.
The worker sees the writable candidate output directory, but not verifier
tests, logs, or reward files. Its IPC uses a strict numeric text format, never
RDS or ``eval`` of candidate-controlled bytes.
"""

from __future__ import annotations

from pathlib import Path


TASK_ID = "terminal-bench-science/duan-thesis"
TASK_SLUG = "duan-thesis"
PROFILE_NAME = "science_duan_profile.R"
WORKER_NAME = "science_duan_worker.R"
LAUNCHER_NAME = "science_duan_mount_exec.py"
R_SCRIPT_WRAPPER = "Rscript"


def profile_source() -> str:
    return r'''local({
if (identical(Sys.getenv("RELAY_DUAN_WORKER"), "1")) return()

.duan_base_source <- base::source
.duan_manager <- new.env(parent = emptyenv())
.duan_manager$ipc <- NULL
.duan_manager$sequence <- 0L

.duan_write_atomic <- function(path, lines) {
  tmp <- paste0(path, ".tmp-", Sys.getpid())
  writeLines(lines, tmp, useBytes = TRUE)
  if (!file.rename(tmp, path)) stop("Duan adapter could not atomically publish IPC file")
}

.duan_encode_numeric <- function(x) {
  if (!is.numeric(x) || !is.atomic(x) || !is.null(dim(x)))
    stop("Duan candidate IPC accepts numeric vectors only")
  vals <- if (!length(x)) character() else vapply(seq_along(x), function(i) {
    v <- x[[i]]
    if (is.na(v) && !is.nan(v)) return("NA")
    if (is.nan(v)) return("NaN")
    if (is.infinite(v)) return(if (v > 0) "Inf" else "-Inf")
    sprintf("%.17g", as.double(v))
  }, character(1), USE.NAMES = FALSE)
  c(as.character(length(x)), vals)
}

.duan_worker_call <- function(name, pars, precip) {
  ipc <- .duan_manager$ipc
  if (is.null(ipc) || !dir.exists(ipc)) stop("Duan candidate worker is unavailable")
  .duan_manager$sequence <- .duan_manager$sequence + 1L
  id <- sprintf("%08d", .duan_manager$sequence)
  req <- file.path(ipc, paste0("request-", id))
  resp <- file.path(ipc, paste0("response-", id))
  lines <- c(name, .duan_encode_numeric(pars), .duan_encode_numeric(precip))
  .duan_write_atomic(req, lines)
  deadline <- proc.time()[[3L]] + 450
  while (!file.exists(resp) && proc.time()[[3L]] < deadline) Sys.sleep(0.05)
  if (!file.exists(resp)) stop("Duan isolated candidate call timed out")
  answer <- readLines(resp, warn = FALSE)
  unlink(resp)
  if (!length(answer)) stop("Duan isolated candidate returned an empty response")
  if (identical(answer[[1L]], "ERROR")) {
    msg <- if (length(answer) > 1L) .duan_hex_decode(answer[[2L]]) else "candidate error"
    stop(paste("Duan isolated candidate failed:", msg), call. = FALSE)
  }
  if (!identical(answer[[1L]], "OK") || length(answer) < 2L)
    stop("Duan isolated candidate returned an invalid response")
  n <- suppressWarnings(as.integer(answer[[2L]]))
  if (is.na(n) || n < 0L || length(answer) != n + 2L)
    stop("Duan isolated candidate returned a malformed numeric vector")
  vals <- if (n == 0L) numeric() else suppressWarnings(as.numeric(answer[seq.int(3L, n + 2L)]))
  if (length(vals) != n || (n && anyNA(vals) && !any(answer[seq.int(3L, n + 2L)] %in% c("NA", "NaN"))))
    stop("Duan isolated candidate returned invalid numeric text")
  vals
}

.duan_hex_decode <- function(x) {
  if (nchar(x, type = "bytes") %% 2L != 0L || grepl("[^0-9a-f]", x)) return("candidate error")
  pairs <- substring(x, seq.int(1L, nchar(x), 2L), seq.int(2L, nchar(x), 2L))
  rawToChar(as.raw(strtoi(pairs, 16L)))
}

.duan_start_worker <- function() {
  if (!is.null(.duan_manager$ipc)) return(invisible(NULL))
  ipc <- tempfile(pattern = "duan-ipc-")
  dir.create(ipc, mode = "0700")
  .duan_manager$ipc <- ipc
  bundle <- "/opt/relay-isolation"
  # Keep startup diagnostics in verifier-only logs. The nested candidate
  # namespace masks /logs, while the trusted R process can inspect failures.
  worker_log <- "/logs/duan-worker.log"
  bwrap <- "/opt/relay-isolation/bin/duan-bwrap-exec"
  args <- c(
    "--die-with-parent", "--unshare-user", "--uid", "0", "--gid", "0",
    "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net", "--cap-drop", "ALL",
    "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
    "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
    "--ro-bind", "/etc", "/etc", "--ro-bind", "/usr/bin", "/usr/bin",
    "--ro-bind", "/usr/sbin", "/usr/sbin", "--ro-bind", "/usr/lib", "/usr/lib",
    "--ro-bind", "/usr/lib64", "/usr/lib64", "--ro-bind", "/usr/libexec", "/usr/libexec",
    "--ro-bind", "/usr/local", "/usr/local", "--ro-bind", "/usr/share", "/usr/share",
    "--ro-bind", "/usr/include", "/usr/include",
    "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
    "--tmpfs", "/run", "--tmpfs", "/tests", "--tmpfs", "/verifier",
    "--tmpfs", "/logs", "--tmpfs", "/home", "--tmpfs", "/root",
    "--tmpfs", "/sys", "--tmpfs", "/dev/shm", "--dir", "/opt",
    "--dir", "/opt/relay-isolation",
    "--bind", "/results", "/results",
    "--ro-bind", "/results/Duan_models.R", "/results/Duan_models.R",
    "--bind", ipc, "/relay-ipc",
    "--ro-bind", file.path(bundle, "science_duan_worker.R"), file.path(bundle, "science_duan_worker.R"),
    "--tmpfs", "/run/secrets",
    # The verifier launches from `/`; the host-side smoke harness may have a
    # cwd that is deliberately absent in this namespace.
    "--chdir", "/", "--clearenv", "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
    "--setenv", "HOME", "/tmp", "--setenv", "RELAY_DUAN_WORKER", "1",
    "--setenv", "RELAY_DUAN_IPC", "/relay-ipc",
    "--setenv", "R_PROFILE_USER", "/dev/null",
    "--", "/usr/local/bin/Rscript", file.path(bundle, "science_duan_worker.R")
  )
  status <- system2(bwrap, args = vapply(args, shQuote, character(1)),
                    wait = FALSE, stdout = worker_log, stderr = worker_log)
  if (!is.null(status) && length(status) == 1L && status == 127L)
    stop("Duan bwrap worker could not start")
  deadline <- proc.time()[[3L]] + 15
  ready <- file.path(ipc, "worker-ready")
  while (!file.exists(ready) && proc.time()[[3L]] < deadline) Sys.sleep(0.1)
  if (!file.exists(ready)) {
    log <- if (file.exists(worker_log)) tail(readLines(worker_log, warn = FALSE), 30L) else character()
    stop(paste(c("Duan isolated worker did not initialize", log), collapse = "\n"))
  }
  invisible(NULL)
}

.duan_source <- function(file, local = FALSE, ...) {
  target <- tryCatch(normalizePath(as.character(file)[1L], winslash = "/", mustWork = FALSE),
                     error = function(e) as.character(file)[1L])
  if (!identical(target, "/results/Duan_models.R"))
    return(.duan_base_source(file, local = local, ...))
  if (!file.exists(target)) stop("Duan candidate model file is missing")
  .duan_start_worker()
  env <- if (identical(local, FALSE)) .GlobalEnv else if (identical(local, TRUE)) parent.frame() else local
  if (!is.environment(env)) stop("Unsupported source() evaluation environment")
  assign("TWOPAR", function(pars, precip) .duan_worker_call("TWOPAR", pars, precip), envir = env)
  assign("SIXPAR", function(pars, precip) .duan_worker_call("SIXPAR", pars, precip), envir = env)
  invisible(NULL)
}

source <- .duan_source
.duan_old_last <- get0(".Last", envir = .GlobalEnv, inherits = FALSE)
.Last <- function() {
  ipc <- .duan_manager$ipc
  if (!is.null(ipc) && dir.exists(ipc)) {
    .duan_write_atomic(file.path(ipc, "stop"), "stop")
    deadline <- proc.time()[[3L]] + 5
    while (!file.exists(file.path(ipc, "worker-stopped")) && proc.time()[[3L]] < deadline) Sys.sleep(0.05)
  }
  if (is.function(.duan_old_last)) try(.duan_old_last(), silent = TRUE)
}
assign("source", .duan_source, envir = .GlobalEnv)
assign(".Last", .Last, envir = .GlobalEnv)
})
'''


def worker_source() -> str:
    return r'''ipc <- Sys.getenv("RELAY_DUAN_IPC")
if (!nzchar(ipc) || !dir.exists(ipc)) quit(status = 121L)
cat("Duan candidate boundary active; hidden verifier-file visible: ",
    file.exists("/tests/hidden-canary.txt"), "\n", file = stderr())
.write_atomic <- function(path, lines) {
  tmp <- paste0(path, ".tmp-", Sys.getpid())
  writeLines(lines, tmp, useBytes = TRUE)
  if (!file.rename(tmp, path)) stop("IPC publish failed")
}
.read_numbers <- function(lines, pos) {
  if (pos > length(lines)) stop("request is truncated")
  n <- suppressWarnings(as.integer(lines[[pos]]))
  if (is.na(n) || n < 0L || n > 100000L || pos + n > length(lines)) stop("invalid vector length")
  if (n == 0L) return(list(value = numeric(), next_pos = pos + 1L))
  raw <- lines[seq.int(pos + 1L, pos + n)]
  if (any(grepl("[^0-9eE+.-]|^(NA|NaN|Inf|-Inf)$", raw))) stop("invalid numeric request text")
  value <- suppressWarnings(as.numeric(raw))
  if (length(value) != n || anyNA(value)) stop("invalid numeric vector")
  list(value = value, next_pos = pos + n + 1L)
}
.hex <- function(x) paste(sprintf("%02x", as.integer(charToRaw(enc2utf8(x)))), collapse = "")
.encode_output <- function(x) {
  if (!is.numeric(x) || !is.atomic(x) || !is.null(dim(x))) stop("candidate must return a numeric vector")
  vals <- if (!length(x)) character() else vapply(seq_along(x), function(i) {
    v <- x[[i]]
    if (is.na(v) && !is.nan(v)) return("NA")
    if (is.nan(v)) return("NaN")
    if (is.infinite(v)) return(if (v > 0) "Inf" else "-Inf")
    sprintf("%.17g", as.double(v))
  }, character(1), USE.NAMES = FALSE)
  c("OK", as.character(length(x)), vals)
}
env <- new.env(parent = .GlobalEnv)
tryCatch(base::source("/results/Duan_models.R", local = env), error = function(e) {
  .write_atomic(file.path(ipc, "worker-error"), .hex(conditionMessage(e)))
  quit(status = 122L)
})
if (!exists("TWOPAR", env, inherits = FALSE) || !is.function(env$TWOPAR) ||
    !exists("SIXPAR", env, inherits = FALSE) || !is.function(env$SIXPAR)) {
  .write_atomic(file.path(ipc, "worker-error"), .hex("candidate did not define both required functions"))
  quit(status = 123L)
}
.write_atomic(file.path(ipc, "worker-ready"), "ready")
idle_deadline <- Sys.time() + 120
repeat {
  if (file.exists(file.path(ipc, "stop"))) break
  requests <- sort(list.files(ipc, pattern = "^request-[0-9]{8}$", full.names = TRUE))
  if (!length(requests)) {
    if (Sys.time() >= idle_deadline) break
    Sys.sleep(0.05)
    next
  }
  request <- requests[[1L]]
  id <- sub("^request-", "", basename(request))
  response <- file.path(ipc, paste0("response-", id))
  tryCatch({
    lines <- readLines(request, warn = FALSE)
    if (!length(lines) || !lines[[1L]] %in% c("TWOPAR", "SIXPAR")) stop("invalid function name")
    pars <- .read_numbers(lines, 2L)
    precip <- .read_numbers(lines, pars$next_pos)
    if (precip$next_pos != length(lines) + 1L) stop("request has extra fields")
    fn <- get(lines[[1L]], envir = env, inherits = FALSE)
    value <- do.call(fn, list(pars = pars$value, precip = precip$value))
    .write_atomic(response, .encode_output(value))
  }, error = function(e) {
    .write_atomic(response, c("ERROR", .hex(conditionMessage(e))))
  })
  unlink(request)
  idle_deadline <- Sys.time() + 120
}
.write_atomic(file.path(ipc, "worker-stopped"), "stopped")
'''


def mount_launcher_source() -> str:
    """Make nested bwrap's inherited mount tree private, then exec bwrap."""
    return r'''#!/opt/relay-runtime/bin/python3
from __future__ import annotations
import ctypes
import os
import sys

if len(sys.argv) < 2:
    raise SystemExit("usage: science_duan_mount_exec.py BWRAP [ARG ...]")
try:
    lines = open("/proc/self/mountinfo", encoding="utf-8").read().splitlines()
except OSError as exc:
    raise SystemExit(f"cannot inspect verifier mount namespace: {exc}")
needs_private = False
for line in lines:
    fields = line.split()
    if len(fields) > 6 and fields[4] == "/":
        separator = fields.index("-")
        needs_private = "unbindable" in fields[6:separator]
        break
if needs_private:
    libc = ctypes.CDLL(None, use_errno=True)
    flags = (1 << 18) | (1 << 14)  # MS_PRIVATE | MS_REC
    if libc.mount(None, b"/", None, ctypes.c_ulong(flags), None) != 0:
        error = ctypes.get_errno()
        raise SystemExit(f"cannot make verifier mount namespace private: errno={error}")
os.execv(sys.argv[1], sys.argv[1:])
'''


def install_bundle(output_dir: Path | str) -> Path:
    """Write the R profile/worker and PATH wrapper into a verifier bundle."""
    output = Path(output_dir)
    (output / PROFILE_NAME).write_text(profile_source())
    (output / WORKER_NAME).write_text(worker_source())
    launcher = output / LAUNCHER_NAME
    launcher.write_text(mount_launcher_source())
    launcher.chmod(0o755)
    wrapper = output / "bin" / R_SCRIPT_WRAPPER
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    bwrap_wrapper = output / "bin" / "duan-bwrap-exec"
    bwrap_wrapper.write_text(
        "#!/bin/sh\n"
        "exec /opt/relay-runtime/bin/python3 /opt/relay-isolation/" + LAUNCHER_NAME + " /opt/relay-tools/bwrap \"$@\"\n"
    )
    bwrap_wrapper.chmod(0o755)
    wrapper.write_text(
        "#!/bin/sh\n"
        "if [ \"${RELAY_SCIENCE_TASK_ID:-}\" = \"" + TASK_ID + "\" ] && [ \"${RELAY_DUAN_WORKER:-}\" != 1 ] && [ \"${1:-}\" = /tests/test_state.R ]; then\n"
        "  export R_PROFILE_USER=/opt/relay-isolation/" + PROFILE_NAME + "\n"
        "fi\n"
        "exec /usr/local/bin/Rscript \"$@\"\n"
    )
    wrapper.chmod(0o755)
    return wrapper

