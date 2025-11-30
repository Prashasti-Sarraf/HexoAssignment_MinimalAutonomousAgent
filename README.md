# Minimal Autonomous Agent - README

## One-liner to run the agent
```bash
python minimal_autonomous_agent.py --dataset_dir spooky-author-identification --output submission.csv --seeds 42 1337 999 --time_limit 86400
```

This autonomous agent inspects the dataset folder to infer its modality (text, tabular, image, or time-series). It identifies key features such as column types, file formats, and sample content to determine the appropriate baseline strategy.

For text datasets, the agent uses TF-IDF vectorization combined with Logistic Regression. For tabular data, it applies Random Forests with simple imputation and scaling. For images, a small 3-layer CNN is employed. These choices are conservative, robust, and require minimal hyperparameter tuning, ensuring reproducibility across diverse MLEbench tasks without dataset-specific hardcoding.

The agent emphasizes auditability: it logs dataset inspection, model selection, seed runs, and validation metrics in `run_log.txt` and `metrics.json`, and outputs a fully autonomous `submission.csv`.

Future self-improvement paths include integrating lightweight AutoML to select models and hyperparameters per modality, leveraging pretrained transformers or CNN backbones for better feature extraction, and implementing adaptive data preprocessing pipelines. These enhancements would increase predictive performance while maintaining the autonomous, one-command execution requirement.

All generated files: logs, metrics, and predictions, are fully self-contained, enabling reproducibility and transparent evaluation.