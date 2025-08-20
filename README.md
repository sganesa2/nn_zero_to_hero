# Neural Networks: Zero to Hero


This repository documents my early experiments with neural networks and my journey to understand their inner workings. The code and concepts here are heavily inspired by Andrej Karpathy's YouTube series, **Neural Networks: Zero to Hero**. Thanks for the wonderful series Andrej!

**NOTE: This repo only contains implementation till makemore Part 2: MLP. I've implemented the rest of the series in individual repositories. Links to them are below.**

---

**Part 1: building micrograd**

- Within this repo

---

**Part 2: building makemore Part 1: Single-layer ngram character-level language model**

- Within this repo

---

**Part 3: building makemore Part 2: MLP ngram character-level language model**

- Within this repo

---

**Part 4: building makemore Part 3: Activations & Gradients, BatchNorm**

- [BatchNormalized MLP Github repo](https://github.com/sganesa2/batchnormalized-mlp)

---

**Part 5: Building makemore Part 4: Becoming a Backprop Ninja**

- [Decoding Backpropagation Github repo](https://github.com/sganesa2/decoding-backpropagation)

---

**Part 6: Building makemore Part 5: Building WaveNet**

- [WaveNet(2016) implementation Github repo](https://github.com/sganesa2/wavenet)

---


**Part 7: Let's build GPT: from scratch, in code, spelled out.**

- Upcoming..

---

**Part 8: Let's build the GPT Tokenizer**

- Upcoming..

---

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
- ['matplotlib'](https://pypi.org/project/matplotlib/) >=3.10.5
- ['torch'](https://pypi.org/project/torch/) >=2.8.0

You can install dependencies using your preferred tool:

```sh
# Using pyproject.toml:
pip install .
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
