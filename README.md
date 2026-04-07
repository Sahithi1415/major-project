# Quantum-Assisted LSTM-Transformer for Cross-Lingual NLP

English to Hindi and English to Telugu neural machine translation using a hybrid architecture:
- Quantum embedding block (PennyLane-based, with classical fallback)
- BiLSTM contextual encoder
- Transformer encoder-decoder for sequence generation

## Project Structure

```text
majorPROJECT/
  data/
    sample_parallel.csv
  src/
    __init__.py
    config.py
    data.py
    model.py
    train.py
    evaluate.py
    inference.py
    utils.py
  docs/
    report_outline.md
  requirements.txt
  README.md
```

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Data Format

Use a CSV file with these columns:
- `source` (English sentence)
- `target` (Hindi or Telugu sentence)
- `target_lang` (`hi` or `te`)

Example is provided at `data/sample_parallel.csv`.

## 3) Train

```bash
python -m src.train --data_path data/sample_parallel.csv --epochs 10 --batch_size 8
```

This saves:
- `artifacts/tokenizer.json`
- `artifacts/model.pt`
- `artifacts/config.json`

## 4) Evaluate

```bash
python -m src.evaluate --data_path data/sample_parallel.csv --checkpoint artifacts/model.pt
```

## 5) Inference

```bash
python -m src.inference --checkpoint artifacts/model.pt --text "How are you?" --lang hi
python -m src.inference --checkpoint artifacts/model.pt --text "Education is important." --lang te
```

## Notes

- Quantum layer uses PennyLane if installed.
- If PennyLane is unavailable, code automatically falls back to a classical projection layer so the project remains runnable.
- For final submission, train with a larger parallel corpus (IIT Bombay EN-HI, Samanantar, OPUS, etc.).

