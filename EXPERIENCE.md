# Independent applied machine learning project

**Sensor Event Detection Under Channel Dropout** · Python, PyTorch, scikit-learn, NumPy, Matplotlib, Model Evidence

- Built a 1,357-parameter CPU PyTorch CNN and compared it with prevalence and logistic baselines across 1,800 held-out synthetic sensor windows from 30 devices. Under sustained channel loss, achieved 0.517 average precision and 0.170 Brier score versus 0.352 and 0.201 for logistic regression.
- Used separate calibration and policy devices to freeze a confidence rule before test. Under sustained loss, the rule issued 79.9% of labels with 19.1% error on issued labels versus 23.9% at full coverage. Quantified review workload, remaining false negatives, and cost sensitivity without changing the threshold.
- Designed group and time holdouts, audited preprocessing boundaries, verified 16,200 prediction rows, and reproduced results in a fresh CPU run. Wrote a technical report and decision memo with the evidence and criteria needed before field evaluation.
