# scripts/70_rebuild

Drivers for the experiments of the paper, "The Trace Is the State: Exact Credit
Assignment for LLM Agent Teams" (arXiv:2603.06859v3). A driver emits, and
optionally runs, the GPU commands of one experiment; the analysis that turns
their output into the numbers of a table lives in `c3/analysis/rebuild/` and
runs on the CPU. No result or dataset ships with the repository.

Every driver has a `--dry-run` mode that prints its commands and starts
nothing, and a CPU unit test under `tests/unit/`.

## Code names and the paper

The file names carry the code names the experiments had while they were run.
This is what each one is in the paper.

| Code name | Paper | Generation (GPU) | Analysis (CPU) |
|---|---|---|---|
| E1 | Section 3; Table 1 (Measured column), Table 3 middle block, Figures 2 and 3; Appendix B.5 (deep workflows, Qwen3-8B) | `e1_cells.py` | `c3.analysis.rebuild.aggregate_e1` |
| E1, signal and noise | Table 1 (Implied column, S / (S + 2N)), Table 3 lower block (S and N) | the E1 cells | `c3.analysis.rebuild.e1_signal_noise` |
| E1b | Table 3 upper block (the noise law on the 3 x 3 grid of n and c) | `e1b_cells.py` | `c3.analysis.rebuild.aggregate_e1b` |
| E3a | Section 4 and Appendix B.6 (ablation bias at each decision point), Table 9 | `e3a_cells.py` | `c3.analysis.rebuild.aggregate_e3a`, `c3.analysis.rebuild.e3a_strata` (Table 9) |
| E4 | Section 4 and Appendix B.7 (learned-critic comparison) | `e4_reference_cells.py`, `e4_critic.py train` and `score` | `e4_critic.py split`, `prepare-reference`, `evaluate` |
| 5-seed run | Section 5 and Table 2; Appendix B.8 and Table 10 (significance) | `scripts/40_train/paper_train.sh` with `RECIPE=main`, then `final_eval.py` | `c3.analysis.rebuild.aggregate_5seed` |
| 3-agent 5-seed run | Appendix B.4, Table 8 | `scripts/40_train/paper_train.sh` with `RECIPE=a3`, then `final_eval.py` with `configs/tasks/math_final_eval_a3.yaml` | `c3.analysis.rebuild.aggregate_5seed` |
| E2 | Section 5 and Appendix B.4 (the long-generation run), Table 7 | `scripts/40_train/paper_train.sh` with `RECIPE=long`, then `final_eval.py`; `export_tag_hf.py` for the step-250 row | `c3.analysis.rebuild.aggregate_e2`, with the training-token account of `c3.analysis.rebuild.e2_tokens` |

The hyperparameters of the three training recipes are Tables 5 and 6 and
Appendix B.2, in `c3/utils/paper_train_contract.py`.

The aggregations write only the result keys their `--manifest` defines. No
manifest with values ships with the repository; `configs/results_keys_v3.json`
lists every key the paper prints, with its unit and the table, figure or
section it appears in, and no value, and is accepted as `--manifest` as it is.

## Commands

The paths below are examples. `20_data/results` is the results root; pass an
absolute path when running for real.

### Section 3: E1 and E1b

```bash
python scripts/70_rebuild/e1_cells.py  --model_root /models --results_root /abs/20_data/results
python scripts/70_rebuild/e1b_cells.py --model_root /models --results_root /abs/20_data/results

python -m c3.analysis.rebuild.e1_signal_noise --results /abs/20_data/results/E1 \
    --out /abs/20_data/results/E1/signal_noise_summary.json \
    --e1-summary /abs/20_data/results/E1/summary.json
```

`e1b_cells.py` runs seed A of a grid point before its seed B, and seed B reuses
the alternatives seed A recorded, so the two differ by continuation noise only.

### Section 4: E3a and E4

```bash
python scripts/70_rebuild/e3a_cells.py --model_root /models --results_root /abs/20_data/results
python -m c3.analysis.rebuild.e3a_strata --results /abs/20_data/results/E3a/a3/4b \
    --out /abs/20_data/results/E3a/strata_summary.json
```

E3a reads the first 150 questions of MATH500 by default (`--split MATH500`).

The E4 sequence (split, reference, critic sweep, scoring, evaluation) with the
settings of Appendix B.7 is written out at the top of `e4_critic.py`. The
reference cells are 16 continuations per alternative under replay seeds 3003
and 4004:

```bash
python scripts/70_rebuild/e4_reference_cells.py --prep <prep dir> --model_root /models \
    --results_root /abs/20_data/results
```

### Section 5 and Appendix B.4: training and the evaluation of record

```bash
export PRETRAIN=Qwen/Qwen3-4B-Instruct-2507
RECIPE=main bash scripts/40_train/paper_train.sh   # Table 2
RECIPE=a3   bash scripts/40_train/paper_train.sh   # Table 8
RECIPE=long bash scripts/40_train/paper_train.sh   # Table 7
DRY_RUN=1 RECIPE=long bash scripts/40_train/paper_train.sh   # print the commands only
EXTRA_ARGS="--actor_num_gpus_per_node 4" RECIPE=long bash scripts/40_train/paper_train.sh   # hardware layout flags
```

`EXTRA_ARGS` is appended to every run for the flags of the machine (GPU and
vLLM engine counts, colocation, offloading); the hyperparameters stay in the
recipe.

The 5-seed runs (Tables 2 and 8) are evaluated once per training seed with
greedy decoding under a 512-token generation limit, and the records are laid
out as `<arm>/<method>/train_s<k>/eval_s0/`, with `a2_4b` the 2-agent workflow
and `a3_4b` the 3-agent one:

```bash
for M in MAPPO MAGRPO C3; do m=$(echo $M | tr A-Z a-z); for K in 0 1 2 3 4; do
  python scripts/70_rebuild/final_eval.py --policy ckpt/_runs/paper_main_${M}_math_seed$K/final_hf \
      --method $M --run_id paper_main_${M}_math_seed$K --generate_max_len 512 \
      --k_by_suite MATH500=1,Minerva-Math=1,AMC23=1,AIME24=1,AIME25=1,GSM8K-test=1,CMATH-test=1 \
      --out /abs/records/a2_4b/$m/train_s$K/eval_s0/final_eval_record.json
  python scripts/70_rebuild/final_eval.py --policy ckpt/_runs/paper_a3_${M}_math_seed$K/final_hf \
      --method $M --run_id paper_a3_${M}_math_seed$K --generate_max_len 512 \
      --task configs/tasks/math_final_eval_a3.yaml \
      --k_by_suite MATH500=1,AIME25=1,CMATH-test=1,GSM8K-test=1 \
      --out /abs/records/a3_4b/$m/train_s$K/eval_s0/final_eval_record.json
done; done
python -m c3.analysis.rebuild.aggregate_5seed --results /abs/records \
    --out /abs/records/summary.json --table /abs/records/tables.md
```

`aggregate_5seed` writes the mean and standard deviation over the training
seeds of MATH500, AIME 2025, CMATH, GSM8K and their average (MATH500, CMATH and
GSM8K, with AIME 2025 beside it), and, for the 2-agent workflow, the paired
test of Table 10: C3 minus the baseline with the higher MATH500 mean, paired by
training seed, with Holm over the four benchmarks. The training-token column of
Table 8 is not aggregated.

The long-generation run (Table 7) trains one model per method and evaluates it
five times, under evaluation seeds 0 to 4, with the defaults of
`final_eval.py` (avg@k, k = 16 on AMC23 and AIME, k = 4 on the others, 2048
tokens). The record layout is the one `aggregate_e2` reads:

```bash
for S in 0 1 2 3 4; do
  python scripts/70_rebuild/final_eval.py --policy ckpt/_runs/paper_long_C3_math_seed0/final_hf \
      --method C3 --run_id paper_long_C3_math_seed0 --seed $S \
      --out /abs/20_data/results/E2/a2_4b/c3/train_s0/eval_s$S/final_eval_record.json
done
python -m c3.analysis.rebuild.aggregate_e2 --results /abs/20_data/results/E2 \
    --manifest configs/results_keys_v3.json --out /abs/20_data/results/E2/summary.json
```

Copy the run's `run_config.json`, `budget_ledger.jsonl` and `train.log` into
`train_s0/` beside the evaluation seeds: the training-token account of Table 7
is read from them.

The step-250 row of MAGRPO in Table 7 is the step-250 checkpoint of the MAGRPO
run written out as a model directory and evaluated like a final model under the
method label `MAGRPO_S250`. `RECIPE=long` keeps the checkpoint tag
`global_step250` to the end of every run (`--save_steps 250 --max_ckpt_num 3`;
the reasoning is beside the two flags in `c3/utils/paper_train_contract.py`).
With `ACTOR` the actor checkpoint directory of that run (`_actor` under its
`--ckpt_path`):

```bash
python $ACTOR/zero_to_fp32.py $ACTOR <export dir>/fp32 -t global_step250
python scripts/70_rebuild/export_tag_hf.py --base <the Qwen3-4B-Instruct-2507 snapshot directory> \
    --state <export dir>/fp32 --out <export dir>/final_hf
python scripts/70_rebuild/final_eval.py --policy <export dir>/final_hf --method MAGRPO_S250 --seed 0 \
    --out /abs/20_data/results/E2/a2_4b/magrpo_s250/train_s0/eval_s0/final_eval_record.json
```
