# Bilateral asymmetry analysis pipeline

## Data layout

External acquisition data is accessed via symlinks under `data/external/`:

- `data/external/dynamic/TeLC/{Mouse_ID}/{Experiment_ID}/` — curated tracking and derived data
- `data/external/raw/TeLC/{Mouse_ID}/{Experiment_ID}/` — raw acquisition files (reserved for future TTL alignment)

Create symlinks as an administrator in PowerShell:

```powershell
cd ~/Documents/bilat_asymmetry_analysis/data/external
New-Item -ItemType SymbolicLink -Path "raw" -Target "C:\Users\pmt10\Data\Mel\raw"
New-Item -ItemType SymbolicLink -Path "dynamic" -Target "C:\Users\pmt10\Data\Mel\dynamic"
```

The experiment ledger for TeLC sessions is `data/telc.csv` (`Mouse_ID`, `Experiment_ID`).

## Snakemake pipeline (v1)

The pipeline is **Snakemake-only** for now. It discovers every `.csv` under `whiskers/left/` and `whiskers/right/` for each ledger row, renders a minimal Neuralyzer JSON (base angle only), runs the Neuralyzer CLI, and writes:

```text
data/processed/TeLC/{Mouse_ID}/{Experiment_ID}/{whisker_key}/base_angle.csv
```

`whisker_key` encodes the side and path under `whiskers/{side}/` (e.g. `left__subfolder__tracking`, `right__subfolder__tracking`).

### Prerequisites

Set `NEURALYZER_RUNNER` to the path of `DataManagerPipelineRunner.exe`:

```powershell
[Environment]::SetEnvironmentVariable(
    "NEURALYZER_RUNNER",
    "C:\path\to\DataManagerPipelineRunner.exe",
    "User"
)
```

### Run

```bash
uv sync
uv run snakemake --cores 1
```

Dry-run the workflow graph:

```bash
uv run snakemake -n
```

Run one job directly (for debugging):

```bash
uv run python pipeline/run_telc_neuralyzer.py \
  --mouse-id Animal_1 \
  --experiment-id Day7 \
  --whisker-key MY_KEY \
  --whisker-csv data/external/dynamic/TeLC/Animal_1/Day7/whiskers/left/sub/tracking.csv \
  --dry-run
```

Sessions must have whisker CSVs on disk before Snakemake will include them in `rule all`. Re-run Snakemake after adding new sessions or whisker files.

## Figures (opt-in)

The bilateral whisker angle stack figure loads processed `base_angle.csv` files and plots a left/right stack for a YAML-named example session.

```bash
uv run snakemake lab_meeting_100926_telc_angle --cores 1
```

Output: `results/figures/lab_meeting_100926/telc_angle/bilateral_stack.pdf`

Options live in [`workflow/figures/lab_meeting_100926/telc_angle/figure_options.yaml`](workflow/figures/lab_meeting_100926/telc_angle/figure_options.yaml). Example sessions (`examples`) are separate from panel layout/plot styling (`panels`).

## Key files

| Path | Role |
|------|------|
| `Snakefile` | Workflow entry point |
| `data/telc.csv` | TeLC session ledger |
| `pipeline/neuralyzer_registry.yaml` | Neuralyzer transform definitions |
| `pipeline/neuralyzer_templates/` | JSON templates |
| `pipeline/run_telc_neuralyzer.py` | Render + run Neuralyzer for one whisker |
| `workflow/processing/scripts/telc_jobs.py` | Ledger + whisker discovery for Snakemake |
| `figures/loaders/base_angle_csv.py` | Reusable loader for Neuralyzer `base_angle.csv` |
| `workflow/figures/lab_meeting_100926/telc_angle/` | Lab meeting TeLC angle stack figure (YAML + driver + smk) |
