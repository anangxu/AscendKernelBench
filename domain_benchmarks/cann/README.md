# CANN domain tasks

Phase 1 defines this suite and its loading, prompt, run-directory, and
evaluation plumbing. No real CANN operator task ships yet: the only exercised
task is the offline fixture under `tests/fixtures/domain_suite/cann/`.

A task directory holds `task.py` with `Model`, `get_inputs`, and
`get_init_inputs`, plus an optional `meta.yaml` with `source`,
`source_revision`, and `category`. The full task id is the directory path
under this root, prefixed with `cann/`, for example `cann/nn/rmsnorm`.
