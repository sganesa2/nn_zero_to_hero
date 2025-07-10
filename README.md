# Neural Networks: Zero to Hero


This repository documents my early experiments with neural networks and my journey to understand their inner workings. The code and concepts here are heavily inspired by Andrej Karpathy's YouTube series, **Neural Networks: Zero to Hero**. Thanks for the wonderful series Andrej!

## Project Structure

- `src/`: Source code for neural network experiments and micrograd implementation.
- `tests/`: Unit tests for the project.
- `README.md`: Project overview and documentation.
- `pyproject.toml`: Project configuration and dependencies.

## Dependencies

This project requires **Python 3.11 or higher**.

**Main dependencies:**
- [`ipykernel`](https://pypi.org/project/ipykernel/) >= 6.29.5 (for Jupyter notebook support)

**Development and testing:**
- [`pytest`](https://pypi.org/project/pytest/) >= 8.4.1

You can install dependencies using your preferred tool:

```sh
pip install -r requirements.txt
# or, using pyproject.toml:
pip install .
# or, with [uv](https://github.com/astral-sh/uv) (a fast Python package installer):
uv pip install -r requirements.txt
```

## Getting Started

To run the main script:

```sh
python src/main.py
```

To run tests:

```sh
pytest
```

---

Feel free to explore, experiment, and contribute!