# Thesis Repository [WIP]

_Note: This repository is a Work-In-Progress._

- Behavioral study code: [`/experiment`](./experiment).
- Cognitive model code: [`/model`](./model).
- Data analysis code: coming soon.
- Data used in the study: The raw and preprocessed data can be found in [`/data`](./data).

## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Running Model Simulations](#running-model-simulations)
- [Running Experiment](#running-experiment)
- [Data Analysis](#data-analysis)
- [Support](#support)
- [Citation](#citation)

## Getting Started

### Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/emfrg/multitasking-n-time-perception-study.git
cd multitasking-n-time-perception-study
```

## Prerequisites

This project requires **Python 3.10** specifically to ensure compatibility with PsychoPy running from source and its dependencies.

### Installing Python 3.10

#### macOS

1. **Using Homebrew (recommended):**

   ```bash
   brew install python@3.10
   ```

2. **Using official installer:**
   - Download Python 3.10 from [python.org](https://www.python.org/downloads/release/python-31011/)
   - Run the installer package
   - Verify installation: `python3.10 --version`

#### Windows

1. Download Python 3.10 from [python.org](https://www.python.org/downloads/release/python-31011/)
2. Run the installer
   - **Important:** Check "Add Python 3.10 to PATH"
   - Select "Install Now" or customize as needed
3. Verify installation by opening Command Prompt: `python --version`

## Environment Setup

### Creating and Activating Virtual Environment

#### macOS

1. **Create virtual environment:**

   ```bash
   python3.10 -m venv .venv
   ```

2. **Activate virtual environment:**

   ```bash
   source .venv/bin/activate
   ```

3. **Install dependencies (excl. PsychoPy):**

   ```bash
   pip install -r requirements.txt
   ```

4. **Deactivate when done:**
   ```bash
   deactivate
   ```

#### Windows

1. **Create virtual environment:**

   ```cmd
   python -m venv .venv
   ```

2. **Activate virtual environment:**

   ```cmd
   .venv\Scripts\activate
   ```

3. **Install dependencies (excl. PsychoPy):**

   ```cmd
   pip install -r requirements.txt
   ```

4. **Deactivate when done:**
   ```cmd
   deactivate
   ```

## Running Model Simulations

Ensure your virtual environment is activated before running simulations.

### Single Tasks (N-back, Typing)

```bash
python -m model.simulations.tasks
```

### Complete Experiment

```bash
python -m model.simulations.experiment
```

## Running Experiment

The behavioral experiment requires PsychoPy. Ensure you're using the Python 3.10 virtual environment we created before.
You can also choose to create a new virtual environment with Python 3.10 or run from the PsychoPy Builder.

### Installation and Execution

Assuming you are still inside the .venv virtual environment:

1. **Navigate to experiment directory:**

   ```bash
   cd experiment
   ```

2. **Install PsychoPy:**

   ```bash
   pip install psychopy
   ```

3. **Run the experiment:**
   ```bash
   python multitasking_experiment.py
   ```

_Note: The terminal or IDE may require input monitoring permission on macOS._

### Alternative Method (if installation issues occur)

If you encounter problems with the command-line installation of PsychoPy:

1. Download PsychoPy Builder from [psychopy.org](https://www.psychopy.org)
2. Open PsychoPy Builder
3. Load the experiment script .py file
4. Run directly from the Builder interface

## Data Analysis

_TODO: Data analysis scripts will be added in future updates._

## Support

For issues or questions, please contact the researcher Emmanuel Fragkiadakis at [m.frgdakis@gmail.com](mailto:m.frgdakis@gmail.com)

## Citation

If you use this code or data in your research, please cite:

Fragkiadakis, E. (2025). _Losing track of time: Computational cognitive modeling of prospective timing under sequential multitasking_ [Master’s thesis, Utrecht University]. Utrecht University Student Theses Repository. https://studenttheses.uu.nl/handle/20.500.12932/50349

### BibTeX:

```bibtex
@mastersthesis{fragkiadakis2025losingtrack,
  title={Losing Track of Time: Computational Cognitive Modeling of Prospective Timing Under Sequential Multitasking},
  author={Fragkiadakis, Emmanuel},
  year={2025},
  school={Utrecht University},
  type={Master's thesis},
  url={https://studenttheses.uu.nl/handle/20.500.12932/50349},
  note={Code and data available at: \url{https://github.com/emfrg/multitasking-n-time-perception-study}}
}
```
