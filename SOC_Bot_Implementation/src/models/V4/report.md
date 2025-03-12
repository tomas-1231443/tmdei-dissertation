# Version 4 - Results Analysis

## Metric Explanation

- **Precision:** Ratio of correct positive predictions to all positive predictions made.
- **Recall:** Ratio of correct positive predictions to all actual positives.
- **F1-Score:** The harmonic mean of precision and recall.
- **Support:** The number of instances of each class in the test set.
- **Overall Accuracy:** The total proportion of correct predictions over all instances.

*Note: For Version 4, the dataset was split 60/40 (training/testing).*
*Note: The model was trained with 100 trees in the random forest.*

---

## Priority Prediction Metrics

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
|   0   |   0.80    |  0.78  |   0.79   |  5868   |
|   1   |   0.79    |  0.79  |   0.79   | 12564   |
|   2   |   0.87    |  0.88  |   0.87   | 17491   |
|   3   |   0.57    |  0.38  |   0.46   |  185    |

**Overall Accuracy:** 83%  
**Macro Average:** Precision 0.76, Recall 0.71, F1-Score 0.73  
**Weighted Average:** Precision 0.83, Recall 0.83, F1-Score 0.83

---

## Taxonomy Prediction Metrics

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
|   0   |   0.55    |  0.25  |   0.35   |   526   |
|   1   |   0.89    |  0.88  |   0.89   |  1565   |
|   2   |   0.94    |  0.95  |   0.94   |  7934   |
|   3   |   0.68    |  0.54  |   0.60   |  1442   |
|   4   |   0.72    |  0.80  |   0.76   |  8124   |
|   5   |   0.73    |  0.73  |   0.73   |  4567   |
|   6   |   0.62    |  0.55  |   0.58   |  2560   |
|   7   |   0.86    |  0.90  |   0.88   |  8005   |
|   8   |   0.79    |  0.53  |   0.64   |  1269   |
|   9   |   0.66    |  0.40  |   0.49   |   116   |

**Overall Accuracy:** 80%  
**Macro Average:** Precision 0.74, Recall 0.65, F1-Score 0.69  
**Weighted Average:** Precision 0.80, Recall 0.80, F1-Score 0.80

---

## Key Analysis Points

- **Data Pre-processing**
  Best implementation of the pre-processing function so far. 
  See `data\cleaned_alerts.csv` for more information.

- **Data Split (60/40):**  
  Version 4 uses a 60/40 split, which provides a larger test set (support totals higher than in V3). This means the training set is slightly smaller than in V3, potentially impacting the model's ability to learn minority classes.

- **Priority Performance:**  
  - Overall Priority accuracy is 83% (slightly lower than V3's 84%).  
  - Underrepresented Class 3 continues to show lower performance.
  
- **Taxonomy Performance:**  
  - Overall Taxonomy accuracy remains at 80%—similar to V3.  
  - Minor differences in class-level metrics, likely due to the different split ratio affecting support counts.

---

## Comparison to Version 3

- **Split Ratio Impact:**  
  - **Version 3** used a 70/30 split, yielding a smaller test set.  
  - **Version 4** uses a 60/40 split, which provides a larger test set.  
    - This slightly reduced the overall Priority accuracy (84% in V3 vs. 83% in V4) likely due to the reduced training data.
  
- **Class-Level Observations:**  
  - For Priority, the performance of majority classes is consistent between versions, but the minority class (Class 3) remains a challenge in both splits.  
  - Taxonomy metrics are very similar between V3 and V4, suggesting that the changes in data split have a marginal impact on these predictions.

- **Overall Conclusion:**  
  The change in the split ratio from 70/30 (V3) to 60/40 (V4) has resulted in a slight drop in Priority accuracy while Taxonomy accuracy remains steady. This implies that while a larger test set gives a more robust evaluation, the smaller training set in V4 may slightly hinder the model's performance on certain classes—particularly those with less representation.

---

## Conclusion

Version 4’s results show that using a 60/40 split leads to an overall Priority accuracy of 83% and Taxonomy accuracy of 80%. Compared to Version 3, the minor drop in Priority performance suggests that a smaller training set can affect the learning of minority classes. Future work may focus on balancing the classes and possibly experimenting with different split ratios or advanced sampling methods to further enhance performance.
