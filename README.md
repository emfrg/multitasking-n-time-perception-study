# Thesis Repository [WIP]
NOTES: This repository is a Work-In-Progress. 
- Code sharing: The full model and experiment code are uploaded. The analysis scripts are coming soon. 
- Data sharing: Please find the raw and preprocessed data used in the study.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Running Model Simulations](#running-model-simulations)
- [Running Behavioral Experiment](#running-behavioral-experiment)
- [Data Analysis](#data-analysis)

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
3. Verify installation by opening Command Prompt:
   ```cmd
   python --version
   ```

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

3. **Install dependencies:**
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

3. **Install dependencies:**
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

#### macOS
```bash
python -m model.simulations.tasks
```

#### Windows
```cmd
python -m model.simulations.tasks
```

### Complete Experiment

#### macOS
```bash
python -m model.simulations.experiment
```

#### Windows
```cmd
python -m model.simulations.experiment
```

## Running Behavioral Experiment

The behavioral experiment requires PsychoPy. Ensure you're using the Python 3.10 virtual environment we created before. 
You can also choose to create a new virtual environment with Python 3.10 or run from the PsychoPy Builder.

### Installation and Execution

Assuming you are still inside .venv virtual environment

#### macOS

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

#### Windows

1. **Navigate to experiment directory:**
   ```cmd
   cd experiment
   ```

2. **Install PsychoPy:**
   ```cmd
   pip install psychopy
   ```

3. **Run the experiment:**
   ```cmd
   python multitasking_experiment.py
   ```

### Alternative Method (if installation issues occur)

If you encounter problems with the command-line installation:

1. Download PsychoPy Builder from [psychopy.org](https://www.psychopy.org)
2. Open PsychoPy Builder
3. Load the experiment script .py file
4. Run directly from the Builder interface

### Participant Resources

Instructions distributed to participants and required resources are available in:
```
resources/qualtrics_survey/
```

## Data Analysis

*TODO: Data analysis scripts will be added in future updates.*

## Troubleshooting

### Common Issues

1. **Python version conflicts:** Ensure you're using Python 3.10 specifically
2. **PsychoPy installation errors:** Try the alternative Builder method
3. **Virtual environment not activating:** Check that Python 3.10 is properly installed
4. **Module not found errors:** Ensure all dependencies are installed in the active virtual environment

### Support

For issues or questions, please open an issue in the repository or contact the maintainer.

## License

[Include your license information here]

## Citation

[Include citation information for your thesis here]
