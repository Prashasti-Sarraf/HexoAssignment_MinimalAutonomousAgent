# Minimal Autonomous Agent - README

## One-liner to run the agent

```bash
python minimal_autonomous_agent.py --dataset_dir <dataset> --output submission.csv --seeds 0 1 2 --time_limit 86400
```
### Overview
This autonomous agent inspects the dataset folder to infer its **modality** (text, tabular, text-seq2seq, image or time-series) by analyzing column types, file formats, and sample content. It then selects a **baseline strategy** suited to the modality:

- **Text:** TF-IDF vectorization + Logistic Regression

- **Text-Seq2Seq / Normalization:** Lightweight character-level GRU or rule-based normalization
(fast and memory-efficient; GRU/T5/Transformer can be used with accuracy-latency trade-offs)

- **Tabular:** LightGBM with simple imputation and categorical handling

- **Image:** Small CNN / ResNet18 backbone
- **Time-Series:** Audio pipeline with 1-channel CNN on Mel spectrograms


The agent emphasizes **auditability** and **reproducibility**. It logs dataset inspection, model choice, seed runs, and validation metrics in `run_log.txt` and `metrics.json`, and outputs a fully autonomous `submission.csv`.

### Experimental Results (3 seeds)
| MLEbench Lite Dataset                  | Modality     | Mean Val Accuracy / Exact Match | SE     | Any Medal (%) |
| -------------------------------------- | ------------ | ------------------------------- | ------ | ------------- |
| Spooky Author Identification           | Text         | 0.8152                          | 0.0029 | 81.52 ± 0.29  |
| Text Normalization Challenge (Seq2Seq) | Text-Seq2Seq | 0.6880                          | 0.0028 | 68.80 ± 0.28  |
| TPS May 2022                           | Tabular      | 0.8404                          | 0.0005 | 84.04 ± 0.05  |
| SIIM-ISIC Melanoma                     | Image        | 0.7423                          | 0.0123 | 74.23 ± 1.23  |
| The ICML 2013 Whale Challenge          | Time-Series  | 0.8421                          | 0.0045 | 84.21 ± 0.45  |


### Internal Workflow
- **Dataset folder** → provide your dataset directory.

- **Modality Detection** → agent analyzes columns, file formats, and sample content.

- **Pipeline Selection** → chooses the appropriate strategy: Text, Text-Seq2Seq, Tabular, Image or Time-Series.

- **Train & Validate Models** → trains baseline model(s) and evaluates using validation set

- **Generate `submission.csv`** → produces predictions for test data.

- **Log metrics** → writes **`metrics.json`** and **`run_log.txt`** for reproducibility.

### Future Improvements
- Integrate lightweight AutoML for model and hyperparameter selection.

- Leverage pretrained embeddings or CNN/backbones for better feature extraction.

- Implement adaptive preprocessing pipelines to boost predictive performance while maintaining one-command execution.

