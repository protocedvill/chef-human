# Capability benchmark

Chef Human includes an opt-in end-to-end benchmark that asks the real configured model to modify a
fresh disposable workspace, then verifies the result outside the agent loop. It is intended to
compare model/program behavior over time—not to produce a universal intelligence score.

## Cases

The levels are cumulative:

| Level | Case | Capability exercised |
| --- | --- | --- |
| `smoke` | `hello_world` | Plan, create one Python file, run it, and finish cleanly |
| `core` | `slugify_contract` | Read a specification and protected tests, implement an API, and run tests |
| `stretch` | `inventory_refactor` | Diagnose existing code, repair behavior, add a second module, and run a multi-file suite |

A case passes only when all three signals are green:

1. Chef Human exits successfully and reports a successful structured result.
2. A separate deterministic command verifies the generated workspace.
3. Seeded specifications and tests marked as protected are byte-for-byte unchanged.

This distinction matters: “the model said it finished” is not treated as proof that the program
works, and modifying a supplied test to make it pass is not counted as success.

## Running it

Prerequisites are the same as a real agent run: a supported Python environment, a running configured
backend, and a model already available locally. The benchmark never downloads a model.

```bash
# List cases without contacting a backend
python scripts/run_benchmark.py --list

# Fastest real end-to-end check (default)
python scripts/run_benchmark.py --through smoke

# Run smoke + core, selecting a model explicitly
python scripts/run_benchmark.py --through core --model qwen2.5-coder:7b

# Run every level and retain artifacts/logs for diagnosis
python scripts/run_benchmark.py --through all --keep-workspaces \
  --output benchmark-report.json
```

Use `--case CASE_ID` to run one case, `--timeout SECONDS` to change the per-case agent timeout, and
`--json` for a versioned machine-readable report on stdout. `--work-dir PATH` retains workspaces at
an explicit location. A failed run returns exit code 1.

## Interpreting results

Record the model name, Chef Human commit, hardware, backend, and settings when comparing runs. Local
model output is nondeterministic even at low temperature, and runtime varies dramatically with
hardware. A useful project history is a series of reports under controlled conditions—not one
headline score.

Each retained case directory contains `agent.log`, the generated files, and any session state. The
JSON report records pass/fail components, duration, agent steps, token usage when supplied by the
backend, changed files, and the tail of verifier output.

A timeout can still show `verifier_success: true`: that means the generated program worked, but the
agent did not complete its own task protocol in time. The case remains a failure because clean
completion is part of the capability being measured.

## Safety boundary

The harness uses fresh directories and subprocess timeouts, but it executes both the agent's shell
commands and its generated code with the current user's permissions. This is not process isolation.
For untrusted models or prompts, run the benchmark inside an external container or disposable VM.
