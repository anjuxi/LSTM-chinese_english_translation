# LSTM English–Chinese Machine Translation

A neural machine translation system built on an LSTM sequence-to-sequence (Seq2Seq) architecture. It translates English into Chinese. The encoder is a bidirectional LSTM and the decoder is a single-direction LSTM, with an attention mechanism to improve translation quality.

## Technical Highlights

- **Encoder**: bidirectional LSTM (BiLSTM), captures context from both directions
- **Decoder**: single-direction LSTM with attention
- **Attention**: global attention, lets the decoder focus on different parts of the source sequence
- **Embeddings**: learnable embedding layer, dimension 256
- **Data processing**: Chinese tokenization with jieba; vocabulary filtering and special tokens

## Project Structure

```
LSTM-chinese_english_translation/
├── DATA_LICENSE.md        # dataset license
├── LICENSE                # project license
├── LSTM-机器翻译.ipynb     # main notebook
├── README.md              # this file
└── data.xls               # English-Chinese parallel corpus
```

## Requirements

See `LSTM-机器翻译.ipynb` for the exact environment and dependencies.

## Model Parameters

| Parameter | Value |
| -------------- | ----- |
| Embedding dimension | 256 |
| Hidden dimension | 512 |
| LSTM layers | 2 |
| Dropout | 0.3 |
| Batch size | 64 |
| Minimum word frequency | 2 |
| Max vocabulary size | 50000 |

## Usage

1. Open `LSTM-机器翻译.ipynb`.
2. Run the cells in order.
3. The notebook loads the data, trains the model, and evaluates it.

## Dataset

Training data is `data.xls`, containing English-Chinese parallel sentence pairs.

Source: `https://www.kaggle.com/datasets/thespringcomes/translationcmn3`

## Training Notes

- GPU acceleration (CUDA) supported
- Fixed random seed for reproducibility
- Model save/load supported
- Training progress is visualized

## Author

**Dr. Ailan Anjuxi (谙弆悕)**

- Email: anjuxi.ME@outlook.com
- SIP: sip:anjuxi@sip.linphone.org

---

*For learning and research use only.*
