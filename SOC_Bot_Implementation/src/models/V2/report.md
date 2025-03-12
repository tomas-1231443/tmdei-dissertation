# Version 2 - Results Analysis

## Metric Explanation

- **Precision:** The ratio of correct positive predictions to all positive predictions made.
- **Recall:** The ratio of correct positive predictions to all actual positives.
- **F1-Score:** The harmonic mean of precision and recall.
- **Support:** The number of actual occurrences of the class in the test set.
- **Overall Accuracy:** The proportion of total correct predictions over all instances.

*Note: For Version 2, the dataset was split 70/30 into training and testing sets.*
*Note: The model was trained with 100 trees in the random forest.*

---

## Priority Prediction Metrics

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
|   0   |   0.76    |  0.71  |   0.73   |  4330   |
|   1   |   0.79    |  0.78  |   0.78   |  9503   |
|   2   |   0.84    |  0.87  |   0.85   | 13108   |
|   3   |   0.64    |  0.39  |   0.48   |   140   |

**Overall Accuracy:** 81%  
**Macro Average:** Precision 0.76, Recall 0.69, F1-Score 0.71  
**Weighted Average:** Precision 0.81, Recall 0.81, F1-Score 0.81

---

## Taxonomy Prediction Metrics

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
|   0   |   0.63    |  0.26  |   0.36   |   387   |
|   1   |   0.85    |  0.89  |   0.87   |  1169   |
|   2   |   0.94    |  0.95  |   0.95   |  5947   |
|   3   |   0.67    |  0.52  |   0.59   |  1068   |
|   4   |   0.72    |  0.79  |   0.75   |  6113   |
|   5   |   0.73    |  0.73  |   0.73   |  3401   |
|   6   |   0.62    |  0.53  |   0.57   |  1930   |
|   7   |   0.85    |  0.91  |   0.88   |  5993   |
|   8   |   0.80    |  0.50  |   0.61   |   985   |
|   9   |   0.60    |  0.34  |   0.43   |    88   |

**Overall Accuracy:** 80%  
**Macro Average:** Precision 0.74, Recall 0.64, F1-Score 0.68  
**Weighted Average:** Precision 0.80, Recall 0.80, F1-Score 0.80

---

## Key Analysis Points

- **Data Pre-processing**
  Not correctly implemented pre-processing. Some alerts still had some noise especially the custom ones and self-reported alerts. 
  See `data\cleaned_alerts_V1&V2.csv` for more information.

- **Data Split:**  
  The model was trained with a 70/30 split, meaning 70% of the data was used for training and 30% for testing. This provides a larger training set, but the evaluation is on a smaller test set.

- **Priority Performance:**  
  - Overall accuracy is 81%.  
  - Classes with larger support (e.g., Class 2) perform well, whereas the underrepresented class (Class 3) shows significantly lower performance.

- **Taxonomy Performance:**  
  - Overall accuracy is 80%.  
  - Most classes perform well (e.g., Classes 1, 2, 7), but minority classes (e.g., Classes 0 and 9) still have poor recall and F1-scores, suggesting that imbalance remains an issue.

- **Conclusion:**  
  The results for Version 2 are similar to Version 1, indicating that changing the split from 60/40 to 70/30 did not significantly alter overall accuracy. The model performs robustly on well-represented classes, but improvements are needed to address class imbalance and boost performance on minority classes.

---