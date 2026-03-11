# Trading Signal Assistant — Fine-Tuned LLM for BTC/Crypto News Classification

A from-scratch LLM project that fine-tunes a small GPT-style model on financial news headlines
to predict directional price movement (UP / DOWN / NEUTRAL). Built following the architecture
principles from *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

---

## Project Goal

Given a financial news headline or short article snippet, the model outputs a trading signal:

| Label    | Meaning                              |
|----------|--------------------------------------|
| `UP`     | Bullish — price likely to rise       |
| `DOWN`   | Bearish — price likely to fall       |
| `NEUTRAL`| No strong directional signal         |

This is framed as a **text classification task**: the GPT backbone encodes the input, and
a small classification head on top of the final token's representation predicts the label.

---

## Project Structure

```
trading-signal-llm/
│
├── data/
│   ├── raw/                    # Raw downloaded news articles, OHLCV price CSVs
│   ├── processed/              # Cleaned and tokenized datasets ready for training
│   └── labeled/                # Final (text, label) pairs — the training/val/test splits
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA: label distribution, text length stats
│   ├── 02_labeling_pipeline.ipynb      # Align news timestamps with BTC price windows
│   ├── 03_baseline_experiments.ipynb   # TF-IDF + logistic regression baseline
│   ├── 04_finetune_gpt.ipynb           # Main fine-tuning loop and evaluation
│   └── 05_inference_demo.ipynb         # Live inference and signal generation demo
│
├── src/
│   ├── data/
│   │   ├── fetcher.py          # Scripts to pull news from APIs (NewsAPI, CryptoPanic, etc.)
│   │   ├── labeler.py          # Auto-labeler: compares news timestamp to ±N hour price window
│   │   ├── dataset.py          # PyTorch Dataset class for (text, label) pairs
│   │   └── tokenizer.py        # BPE tokenizer wrapper (tiktoken or custom)
│   │
│   ├── model/
│   │   ├── gpt.py              # GPT architecture (from scratch, per Raschka Ch. 4)
│   │   ├── attention.py        # Multi-head self-attention module
│   │   └── classifier.py       # Classification head on top of GPT backbone
│   │
│   ├── training/
│   │   ├── trainer.py          # Training loop with loss, optimizer, scheduler
│   │   ├── loss.py             # Cross-entropy loss + class imbalance handling
│   │   └── callbacks.py        # Checkpointing, early stopping, LR scheduling
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # Accuracy, F1, precision/recall per class
│   │   ├── confusion.py        # Confusion matrix plotting
│   │   └── backtester.py       # Simulated P&L: trade BTC based on model predictions
│   │
│   └── inference/
│       ├── predictor.py        # Load checkpoint and classify a single headline
│       └── batch_predict.py    # Run predictions over a CSV of headlines
│
├── configs/
│   ├── model_small.yaml        # ~10M param GPT config (for fast iteration)
│   ├── model_medium.yaml       # ~50M param GPT config (for best accuracy)
│   ├── finetune.yaml           # Training hyperparameters: LR, batch size, epochs
│   └── data.yaml               # Dataset paths, train/val/test split ratios, label windows
│
├── scripts/
│   ├── download_news.sh        # Bash wrapper to fetch and store raw news data
│   ├── build_dataset.sh        # End-to-end: fetch → label → split → save
│   └── train.sh                # Launch training with a specific config
│
├── tests/
│   ├── test_labeler.py         # Unit tests for the auto-labeling logic
│   ├── test_dataset.py         # Verify dataset __getitem__ shapes and dtypes
│   ├── test_model.py           # Sanity check forward pass shapes
│   └── test_backtester.py      # Verify P&L calculation correctness
│
├── docs/
│   ├── architecture.md         # Diagram and description of the model architecture
│   ├── labeling_strategy.md    # How news is aligned to price windows + edge cases
│   ├── data_sources.md         # List of all data sources with API setup instructions
│   └── results.md              # Experiment log: hyperparams, metrics, observations
│
├── outputs/
│   ├── checkpoints/            # Saved model weights (.pt files)
│   ├── logs/                   # Training loss curves (CSV or TensorBoard format)
│   └── reports/                # Final evaluation reports and backtest P&L charts
│
├── .gitignore
├── requirements.txt
└── README.md                   # This file
```

---

## Methodology

### 1. Data Collection
News headlines and short summaries are collected from financial/crypto news APIs
(e.g., CryptoPanic, NewsAPI, CoinDesk RSS). BTC OHLCV price data comes from Binance or
a similar exchange API.

### 2. Auto-Labeling
Each news item is timestamped. We look at BTC's closing price at `t` and `t + N hours`
(configurable, e.g. 4h or 24h). If price rose by more than a threshold `δ`, label = UP.
If it fell more than `δ`, label = DOWN. Otherwise, NEUTRAL.

```
label = UP     if  price(t+N) / price(t) - 1  >  +δ
label = DOWN   if  price(t+N) / price(t) - 1  <  -δ
label = NEUTRAL otherwise
```

### 3. Model Architecture
A small GPT model (from scratch, following Raschka Ch. 4–6) serves as the text encoder.
The final hidden state of the last token is passed through a linear classification head:

```
Input headline → Tokenizer → GPT Transformer Blocks → [CLS token repr] → Linear(d_model, 3) → Softmax → {UP, DOWN, NEUTRAL}
```

### 4. Fine-Tuning Strategy
- **Base model**: Either pretrained GPT-2 weights (loaded into our architecture) or a small
  model pretrained from scratch on financial corpora.
- **Fine-tuning**: Only the classification head is trained first (frozen backbone), then
  the entire model is unfrozen for full fine-tuning.
- **Class imbalance**: NEUTRAL tends to dominate. Use weighted cross-entropy loss or
  oversample minority classes.

### 5. Evaluation
Beyond standard classification metrics (accuracy, F1), the model is evaluated on a
**simulated backtest**: trade 1 BTC whenever the model predicts UP or DOWN with
confidence above a threshold, and measure cumulative P&L vs. a buy-and-hold baseline.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch and label data
bash scripts/build_dataset.sh

# 3. Train the model
bash scripts/train.sh configs/finetune.yaml

# 4. Run inference on a headline
python src/inference/predictor.py --text "Bitcoin ETF approved by SEC"
# → Predicted: UP (confidence: 0.87)
```

---

## Key Design Decisions & Open Questions

- **Label window size**: 4h vs. 24h — shorter windows are noisier; longer windows may capture
  events unrelated to the specific news item.
- **Threshold δ**: Too small → noisy labels; too large → few UP/DOWN examples.
- **Text length**: Headlines only vs. full article leads — more context may help but increases
  sequence length and training cost.
- **Pretrained backbone**: Starting from GPT-2 weights vs. training from scratch will
  significantly impact data requirements and final accuracy.

---

## Roadmap

- [ ] Baseline: TF-IDF + logistic regression (non-LLM benchmark)
- [ ] Implement GPT classification architecture from scratch
- [ ] Auto-labeling pipeline with configurable `N` and `δ`
- [ ] Fine-tuning loop with frozen → unfrozen training stages
- [ ] Backtesting engine with P&L vs. buy-and-hold comparison
- [ ] LoRA fine-tuning variant (parameter-efficient)
- [ ] Live signal generation from real-time news feed

---

## References

- Raschka, S. — *Build a Large Language Model (From Scratch)* (2024)
- Brown et al. — *Language Models are Few-Shot Learners* (GPT-3 paper)
- [CryptoPanic API](https://cryptopanic.com/developers/api/)
- [Binance Public API](https://binance-docs.github.io/apidocs/)