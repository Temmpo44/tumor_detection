Disclaimer: This repository is for research/demo purposes only and is not a medical device. Clinical decisions must be made by qualified radiologists/physicians.

Overview

Task: 4-class MRI classification — glioma, meningioma, pituitary, notumor

Models: PyTorch (ensemble-friendly), optional YOLO overlay for boxes, optional live DICOM watcher

Environment: Windows-optimized logs and memory handling

Results (Sample)

Accuracy: 0.97

Macro F1: 0.97

Per-class:

glioma: P=0.98, R=0.95, F1=0.96

meningioma: P=0.93, R=0.94, F1=0.93

notumor: P=0.98, R=1.00, F1=0.99

pituitary: P=0.98, R=0.97, F1=0.97
Dataset

Source: Kaggle – Brain Tumor MRI Dataset (4-class) (glioma, meningioma, pituitary, notumor)

Note: Downloaded and used under Kaggle’s terms for research only.
Validation transforms must be deterministic; use shuffle=False, drop_last=False.

Always call model.eval() with torch.no_grad() for evaluation.

Consider temperature scaling + class-wise thresholds to reduce glioma↔meningioma confusions.

Use TTA (flip/rotate) for small but stable gains.

Ensure patient-level splits to prevent leakage.

Ethics & Privacy

Do not include PHI in images.

Model outputs are non-diagnostic decision support.

License

Code: MIT (or your choice)

Data: Subject to Kaggle dataset’s license/terms.

Acknowledgements

Kaggle Brain Tumor MRI Dataset

PyTorch, scikit-learn, Ultralytics YOLO, pydicom, OpenCV
