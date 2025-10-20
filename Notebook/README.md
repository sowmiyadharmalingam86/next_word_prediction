# Next Word Prediction Notebook

Use the commands below to launch the provided notebook with the project’s `uv`-managed environment.

## 1. Sync dependencies

```bash
cd /path/to/next_word_prediction
uv sync
```

This creates (or updates) the virtual environment declared in `pyproject.toml`, installing TensorFlow and the other notebook requirements.

## 2. Start Jupyter Notebook

```bash
uv run jupyter notebook Notebook/Next_Word_Prediction_with_Deep_Learning_in_NLP.ipynb
```

`uv run` ensures the notebook server uses the synced environment. When the browser tab opens, you can execute the cells directly.

> **Tip:** Run the notebook from the project root so the relative paths (for example `../data/raw/sherlock_holmes.txt`) resolve correctly.

## (Optional) Register the Kernel

Jupyter will automatically use the `uv` environment when launched with `uv run`. If you prefer to see a named kernel in the UI, register it once:

```bash
uv run python -m ipykernel install --user --name next-word-prediction
```

Then pick it inside the notebook via `Kernel → Change kernel → next-word-prediction`.
