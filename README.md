**Dynamic Poisson Factorization (PyTorch)**
- Reproduces [Charlin et al., 2015](https://arxiv.org/abs/1509.04640) with a lightweight PyTorch implementation.
- Original C++ code [here](https://github.com/blei-lab/DynamicPoissonFactorization/)
- Data: grab the TSVs from the paper’s repo and drop `train.tsv`, `validation.tsv`, `test.tsv`, `test_users.tsv` into `data/`.

**Setup**
We use uv for the python environment and dependency management. Install uv then run:

```bash
uv sync
```

**Train + evaluate**
```bash
uv run python src/main.py
```
Prints paper metrics and writes plots to `assets/`.
