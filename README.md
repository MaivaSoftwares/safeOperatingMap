# Supplementary Material

**Wake-Recovery Characterization and a Safe Operating Map for Duty-Cycling a Precision
Closed-Loop MEMS Accelerometer**
Mustapha Dauda, Wu Zhou, Zhaoli Wei, Longqi Ran. IEEE Transactions on Instrumentation and
Measurement (under revision).

This bundle lets a reader regenerate the paper's figures and recompute its tables and the numbers
of its validation, comparison and transfer sections from the measured hardware data, through the
same C code the sensor node runs to classify operating points (`core/safemap.c`). It is an extract
of the project's research repository, scoped to this paper.

## What's here

```
core/            safemap.c / safemap.h: the firmware's map engine (guard, energy model,
                 constraint classifier). Built into twin/libcore.so.
twin/bindings/   core_ffi.py: a cffi binding, so the Python scripts call the compiled C.
twin/analysis/   the analysis scripts (entry points below) and the modules they import.
twin/device/     the bench's serial driver, imported for its CSV parser (no hardware needed).
twin/data/       temperature-ramp logs (bias vs die temperature) of the hysteresis campaign.
tools/           the burst tool whose transient fit every burst analysis uses
                 (run_r1_burst_repeatability.py); it is imported, not run.
tools/data/      raw captures and logs, named by unit, campaign and capture time.
paper/figures/   make_revision_figures.py and the generators it imports; rev/ holds its output.
output/          the figures and the printed output of every entry point from a run of this bundle.
```

The folders mirror the research repository, so each script finds its data by the same relative
paths it uses there.

## Build and run

Python 3.12 with the packages in `requirements.txt`, and a C compiler.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make                                   # -> twin/libcore.so from core/safemap.c
.venv/bin/python paper/figures/make_revision_figures.py      # all figures -> paper/figures/rev/
```

## Where each result comes from

Run every command from the bundle's root with the virtual environment's Python (`.venv/bin/python`).
`output/` holds what each command printed in a run of this bundle; the figures land in
`paper/figures/rev/`.

| In the paper | Command | Data read |
|---|---|---|
| Table II, measured rows (mean ± SD) | `twin/analysis/p1_table2.py` | `r1_power_*`, `r1_burst_*_n30_*`, `r1_inrush_*shunt1p00*` |
| Table II guards; Section VII (transfer) | `twin/analysis/p1_transfer_revision.py` | the coefficient sets of Table II, written in the script |
| Fig. 3 (wake-current surge; recovery terms) | `paper/figures/make_revision_figures.py inrush recovery_terms` | `pd_scope/`, `rail_off_scope/`, the 30-burst calibration set |
| Section III-B (one or two recovery terms) | `twin/analysis/p1_recovery_models.py` | the 30-burst calibration sets |
| Fig. 4; Section VIII (hysteresis, r_max, guards that carry the loop) | `paper/figures/make_revision_figures.py hyst_campaign`; `cd twin && ../.venv/bin/python analysis/p1_ramp_guard.py` | `twin/data/temp_log_*_hyst_*.csv` |
| Table V, Fig. 5 (the map and its bands); bandwidth with a saving | `twin/analysis/p1_map_revision.py`, then `--bandwidth`; `paper/figures/make_revision_figures.py safemap_banded` | the Table II runs; writes `tools/data/r3_map_uncertainty_unitA_rev/` |
| Fig. 6 (truncated guards) | `paper/figures/make_revision_figures.py b2_guard_violation` | `unitA_b2_guard_sweep_20260709_082651/` |
| Fig. 7 (thermal collapse) | `paper/figures/make_revision_figures.py thermal_paper` | `ctl_thermal_*.log` (four ramps per unit and the two-ramp control) |
| Fig. 8; Section V-E (real drift) | `twin/analysis/p1_m3_audit.py tools/data/m3_real_unitA_20261016_024528 --delays` (and `..._unitB_20261012_231635`); `paper/figures/make_revision_figures.py m3_drift` | `m3_real_unit*/`, `ctl_m3_real_*`, `r1_burst_repeatability_m3_real_*` |
| Fig. 9 (transfer) | `paper/figures/make_revision_figures.py transfer_paper` | the coefficient sets of Table II |
| Table IV (uncertainty budget of the guard) | `twin/analysis/p1_uncertainty_budget.py` | the calibration sets and the burst record |
| Table VI, Section VI (rules for the first trusted reading) | `twin/analysis/p1_benchmark_revision.py`, with `--fine` (the slope rule's finer grid) and `--moving` (the sensor moving) | the burst record |
| Section III-F (record checks: 4508 bursts, late first valid frames, the start-up rule) | `twin/analysis/p1_revision_checks.py` | the burst record |
| The slow term b_s (Sections II-B, III-B) | `twin/analysis/p1_slow_term_october.py` | the burst record |

`make_revision_figures.py` with no argument draws every figure; it also draws `recovery_raw.pdf`,
which the paper does not use. Fig. 1 is a diagram and Fig. 2 a photograph of the bench.

"The burst record" is every power-down burst captured from July 6 to October 3, 2026: 4508 bursts
in 183 run folders under `tools/data/`, found by the scripts from the folder names and capture
dates (`p1_revision_checks.record_runs`). Many of these runs were taken in deliberately provoked
states for a companion study; the paper uses them only as the record of wakes.

## What's not here, and why

- The logs of the controller campaign (Section V-D). The controller is the subject of a separate
  paper; its code and the logs of its comparisons are in that paper's bundle,
  [github.com/MaivaSoftwares/adaptiveControl](https://github.com/MaivaSoftwares/adaptiveControl).
- The captures behind Section III-D (unit A's flips are here as the per-cycle means the map's
  propagation reads; unit B's are not), Section III-E (the bias temperature coefficient fit) and
  Section V-B (the operating-point grid). Available on request.
- The rest of the research repository: the node's firmware other than `core/safemap.c`, the
  digital twin's engine, dashboard and tests, and the bench's acquisition scripts other than the
  burst tool's fit.

## Data provenance

Every file under `tools/data/` and `twin/data/` is a raw, unedited capture or log from the bench
described in the paper (an STM32L476 node driving an MMS-30 closed-loop accelerometer, an INA219
current monitor, an oscilloscope across a 1.00 ohm shunt), or a file the scripts here write from
them (`r3_map_uncertainty_unitA_rev/`, written by `p1_map_revision.py`). Folder names carry the
unit, the campaign and the capture time. One edit was made for this bundle: file paths that the
acquisition scripts recorded in run summaries and event logs are given relative to the repository
root (one local source folder is written as "(local folder)"); no script reads them.

## License

MIT, see `LICENSE`.

## Contact

Questions about reproducing a result, or requests for raw data not included here, can be directed
to the corresponding author (Wu Zhou) or the e-mail addresses listed in the paper.
