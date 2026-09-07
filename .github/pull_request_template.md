## What changed

- 

## Why this change

- 

## Reproducibility impact

- [ ] No impact on training/eval outputs
- [ ] Affects data preparation outputs (updated `configs/data_manifest.yaml`)
- [ ] Affects evaluation/analysis pipeline behavior

## Risk assessment

- Edge cases:
- Backward compatibility:
- Performance impact:

## How to test

- [ ] `pytest -q tests`
- [ ] `bash scripts/90_audit/pre_release.sh`
- [ ] `bash scripts/30_smoke/smoke.sh --task tests/fixtures/tasks/mini_math.yaml --limit 1 --print_example 0`
- [ ] Additional task-specific commands:

## Reviewer checklist

- [ ] CLI/API compatibility is preserved or explicitly documented
- [ ] Docs/configs updated with behavior changes
- [ ] No secrets, private paths, or large artifacts introduced
- [ ] Repro steps are complete and deterministic
