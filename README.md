# chef-human
Where a real mind is in charge.

A local AI software development tool that runs open-source models on your machine.

This is a Linux-first learning prototype, not a production autonomous agent. Its workspace checks,
shell blacklist, approval prompts, and timeouts reduce common accidents but do not isolate spawned
processes. Run it only in a disposable checkout or another sandbox you control.

## Quick Start

```bash
bash scripts/setup.sh
```

## Documentation

- [Installation](docs/INSTALL.md) — setup, dependencies, troubleshooting
- [Usage](docs/USAGE.md) — API examples, configuration, testing
- [Testing](docs/TESTING.md) — test taxonomy, optional extras, live-backend checks
- [Capability benchmark](docs/BENCHMARKS.md) — tiered real-agent tasks with external verification
