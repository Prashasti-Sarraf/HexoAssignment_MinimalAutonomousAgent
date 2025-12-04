# Minimal Autonomous Agent - README

## One-liner to run the agent

```bash
python minimal_autonomous_agent.py --dataset_dir <dataset> --output submission.csv --seeds 0 1 2 --time_limit 86400
```


### **Overview**

This autonomous agent inspects the dataset folder to infer its **modality** (text, tabular, image or time-series) by analyzing column types, file formats, and sample content. It then selects a **baseline strategy** suited to the modality:

- **Text:** TF-IDF vectorization + Logistic Regression

- **Tabular:** LightGBM with simple imputation and categorical handling
- **Image:** Small CNN / ResNet18 backbone

- **Time-series:** Placeholder pipeline (extendable)

The agent emphasizes **auditability** and **reproducibility**. It logs dataset inspection, model choice, seed runs, and validation metrics in `run_log.txt` and `metrics.json`, and outputs a fully autonomous `submission.csv`.

### Experimental Results (3 seeds)
| MLEbench Lite Dataset                      | Modality | Mean Val Accuracy    | SE     | Any Medal (%) |
| ---------------------------- | -------- | -------------------- | ------ | ------------- |
| Spooky Author Identification | Text     | 0.8152               | 0.0029 | 81.52 ± 0.29  |
| TPS May 2022                 | Tabular  | 0.8404               | 0.0005 | 84.04 ± 0.05  |
| SIIM-ISIC Melanoma           | Image    | Training in progress | -      | -             |

### Internal Workflow
- **Dataset folder** → provide your dataset directory.

- **Modality Detection** → agent analyzes columns, file formats, and sample content.

- **Pipeline Selection** → chooses the appropriate strategy: Text, Tabular, Image or Time-series.

- **Train & Validate Models** → trains baseline model(s) and evaluates using validation set.

- **Generate `submission.csv`** → produces predictions for test data.

- **Log metrics** → writes **`metrics.json`** and **`run_log.txt`** for reproducibility.

### Future Improvements
- Integrate lightweight AutoML for **model and hyperparameter selection**.

- Leverage **pretrained embeddings or CNN/backbones** for better feature extraction.

- Implement **adaptive preprocessing pipelines** to boost predictive performance while maintaining **one-command execution**.
