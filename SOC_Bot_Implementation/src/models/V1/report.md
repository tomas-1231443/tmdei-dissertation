# Version 1 - Results Analysis

## Evaluation Metrics Explanation

- **Precision:** The ratio of correct positive predictions to all positive predictions made.
- **Recall:** The ratio of correct positive predictions to all actual positives.
- **F1-Score:** The harmonic mean of precision and recall.
- **Support:** The number of actual occurrences of the class in the test set.
- **Overall Accuracy:** The proportion of total correct predictions over all instances.

*Note: The dataset was split 60/40 (training/test) for evaluation.*
*Note: The model was trained with 100 trees in the random forest.*

---

## Priority Prediction Metrics

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
|   0   |   0.77    |  0.70  |   0.73   |  5868   |
|   1   |   0.78    |  0.78  |   0.78   | 12564   |
|   2   |   0.84    |  0.87  |   0.85   | 17491   |
|   3   |   0.56    |  0.38  |   0.45   |  185    |

**Overall Accuracy:** 81%  
**Macro Avg:** Precision 0.74, Recall 0.68, F1-Score 0.70  
**Weighted Avg:** Precision 0.80, Recall 0.81, F1-Score 0.80

---

## Taxonomy Prediction Metrics

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
|   0   |   0.61    |  0.25  |   0.36   |  526    |
|   1   |   0.86    |  0.90  |   0.88   |  1565   |
|   2   |   0.94    |  0.95  |   0.94   |  7934   |
|   3   |   0.68    |  0.51  |   0.58   |  1442   |
|   4   |   0.72    |  0.79  |   0.75   |  8124   |
|   5   |   0.73    |  0.72  |   0.73   |  4567   |
|   6   |   0.62    |  0.54  |   0.58   |  2560   |
|   7   |   0.85    |  0.91  |   0.88   |  8005   |
|   8   |   0.77    |  0.50  |   0.61   |  1269   |
|   9   |   0.62    |  0.49  |   0.55   |  116    |

**Overall Accuracy:** 80%  
**Macro Avg:** Precision 0.74, Recall 0.66, F1-Score 0.69  
**Weighted Avg:** Precision 0.80, Recall 0.80, F1-Score 0.79

---

## Key Analysis Points
- **Data Pre-processing**
  Not correctly implemented pre-processing. Some alerts still had some noise especially the custom ones and self-reported alerts. 
  See `data\cleaned_alerts_V1&V2.csv` for more information.

- **Data Split:**  
  The dataset was split 60/40 (training/test). With 90K+ rows, this provides a solid evaluation base, although minority classes still suffer from low support.

- **Priority Prediction:**  
  - Overall accuracy is 81%.
  - Classes with ample data (e.g., class 2) perform well.
  - Class 3 is underrepresented, leading to poor performance (F1 0.45).

- **Taxonomy Prediction:**  
  - Overall accuracy is 80%.
  - Some classes (e.g., class 2, class 1, class 7) have high precision/recall.
  - Minority classes (e.g., classes 0 and 9) show weak performance, suggesting imbalance issues.

- **Conclusion:**  
  The results are promising for majority classes but highlight the challenge of class imbalance. Improvements could include re-sampling strategies, class weighting, and potentially more advanced text representations (e.g., embeddings).
