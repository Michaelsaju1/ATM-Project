# ATM Project

This project uses a LightGBM model to predict if a user will repay their loan in 30 days.

## Requirements

- Python 3.14+
- uv Package Manager
- The `.parquet` dataset files

## Installation

This project uses the uv package manager. It is very easy to get up and running. Create a virtual environment and
install the dependencies first:
```shell
uv venv
uv sync
```

Then, you can run the program with:
```shell
uv run atm_project.py
```