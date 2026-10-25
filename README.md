# Shared representations of educational data

Experiments for *Constructing a Shared Representation of Educational Data under Joint Analytical Task Requirements*, by Denis S. Vedenskiy, HSE University.

The study compares separate and shared aggregate representations on MoocRadar, ALPACAS, and the Moodle Student Interaction Dataset. It measures storage, construction, query response, and update costs, with additional experiments on representation selection, scaling, and selective maintenance.

## Getting started

Create the environment and open JupyterLab from the repository root:

```sh
conda env create -f environment.yml
conda activate educational-analytics
jupyter lab experiment.ipynb
```

`environment.yml` reproduces the Windows Python 3.11 environment used for the measurements. It includes a platform-specific PyArrow build. For other platforms, install compatible builds of the packages in `requirements.txt`, then install `requirements-notebooks.txt`. ClickHouse runs in a Linux container using `compose.yaml`, or on an existing local server of the same version (25.3.2.39).

Open [experiment.ipynb](experiment.ipynb) in the repository root. It covers data preparation, experiment execution, and result analysis in one workflow. Paths and the run mode are set once at the top. With `RESULTS = None`, the notebook prepares missing inputs and runs a new experiment. Set `RESULTS` to an existing result directory to skip preparation and measurement and inspect that run.

Verification of an existing run requires its original prepared inputs and matching source code.

Download the MoocRadar source files and the Moodle ZIP from the links below before preparing the data. ALPACAS is downloaded by its preparation function. Prepared data, credentials, and results stay outside version control.

The notebook imports the Python functions directly. The experiment can also be called from other Python code:

```python
from experiments.run import run_experiment

results = run_experiment(mode="measure", connection_file=".env")
```

`run_experiment` runs each measurement stage in a fresh worker process to isolate it from the notebook kernel. Preparation and analysis execute in the calling Python process. The function returns a `Path` to the completed results and raises an exception on failure.

In a standalone Python script, place the experiment call under `if __name__ == "__main__":` for multiprocessing. Notebook cells can call it directly.

Start ClickHouse before the experiment. To use the bundled service, create `.env` from `.env.example`, set its password, and run `docker compose up -d`.

The same experiment function is available through a command-line entry point:

```sh
python run_experiments.py measure --connection-file .env --out outputs/campaign
```

Use `smoke` instead of `measure` for a short execution check. Run one experiment at a time, with a fresh output directory. Each run fits its own models, selects representations on calibration data, and evaluates them on held-out data. Raw paired timings, input hashes, environment details, and correctness checks are saved alongside the summaries. The primary statistic is the arithmetic mean, with paired moving-block bootstrap confidence intervals. The declared warm-up rounds are excluded from estimates; all completed measurement pairs are retained.

## Code

`adapters/` reconstructs task inputs, `scripts/` prepares public datasets, and `experiments/` implements selection, measurement, maintenance, and statistical analysis. Source fingerprints, observation windows, and preparation settings are in `configs/`. Measurement settings are in `experiments/protocol.json`.

Run `python -m unittest discover -s tests -v` to check identity constraints, half-open windows, window shifts, aggregate reconstruction, and correction bounds. These tests require no database or downloaded data. Selective maintenance supports binary-value replacements on existing events with fixed keys, timestamps, and event counts. Other changes require full reconstruction.

## Reported results

Percentage reductions for the shared representation relative to separate storage of the same selected task inputs:

| Dataset | Prepared storage | Mean answer time | Mean construction time | Mean update time |
|---|---:|---:|---:|---:|
| MoocRadar | 47.9% | 38.7% | 61.3% | 58.0% |
| ALPACAS | 49.1% | 30.4% | 49.7% | 44.3% |
| Moodle | 36.5% | 31.4% | 48.5% | 42.6% |

These values are from the manuscript's measured experiment. The notebook displays results from the selected local run. Joint enumeration selected the same configuration as independent selection followed by union and took 42.0% longer. At replication factor 50, the maintenance comparison did not establish a response-time benefit of the adaptive policy over batch processing.

## Datasets

- **MoocRadar:** [source repository](https://github.com/THU-KEG/MOOC-Radar). Download `student-problem-middle.json` and `problem.json`. J. Yu et al., *MoocRadar: A Fine-grained and Multi-aspect Knowledge Repository for Improving Cognitive Student Modeling in MOOCs*, SIGIR 2023. [DOI](https://doi.org/10.1145/3539618.3591898).
- **ALPACAS:** [Zenodo, version 1](https://doi.org/10.5281/zenodo.17087849). E. Tiukhova, D. Van Landuyt, and M. Snoeck, *Open Data, Private Learners: A De-Identified Dataset for Learning Analytics Research*, 2025.
- **Moodle Student Interaction Dataset:** [Mendeley Data, version 1](https://doi.org/10.17632/x8ntc8skgs.1). L. M. Cedeno Pinargote and C. A. Silva Villafuerte, 2025. Download the dataset as a ZIP.

Dataset access and reuse are governed by their publishers' terms. Third-party packages retain their respective licenses.
