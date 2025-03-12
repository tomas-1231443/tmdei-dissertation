# Version 3 - Results Analysis

## Metric Explanation

- **Precision:** Ratio of correct positive predictions to all predicted positives.
- **Recall:** Ratio of correct positive predictions to all actual positives.
- **F1-Score:** The harmonic mean of precision and recall.
- **Support:** Number of instances of each class in the test set.
- **Overall Accuracy:** Total proportion of correct predictions over all instances.

*Note: For Version 3, the dataset was split 70/30 (training/testing).*
*Note: The model was trained with 100 trees in the random forest.*

---

## Priority Prediction Metrics

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
|   0   |   0.80    |  0.80  |   0.80   |  4330   |
|   1   |   0.80    |  0.80  |   0.80   |  9503   |
|   2   |   0.88    |  0.88  |   0.88   | 13108   |
|   3   |   0.61    |  0.37  |   0.46   |   140   |

**Overall Accuracy:** 84%  
**Macro Average:** Precision 0.77, Recall 0.71, F1-Score 0.73  
**Weighted Average:** Precision 0.84, Recall 0.84, F1-Score 0.84

---

## Taxonomy Prediction Metrics

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
|   0   |   0.56    |  0.28  |   0.37   |   387   |
|   1   |   0.88    |  0.88  |   0.88   |  1169   |
|   2   |   0.94    |  0.95  |   0.95   |  5947   |
|   3   |   0.65    |  0.56  |   0.60   |  1068   |
|   4   |   0.73    |  0.79  |   0.76   |  6113   |
|   5   |   0.73    |  0.74  |   0.74   |  3401   |
|   6   |   0.62    |  0.55  |   0.58   |  1930   |
|   7   |   0.86    |  0.91  |   0.88   |  5993   |
|   8   |   0.80    |  0.53  |   0.64   |   985   |
|   9   |   0.61    |  0.34  |   0.44   |    88   |

**Overall Accuracy:** 80%  
**Macro Average:** Precision 0.74, Recall 0.65, F1-Score 0.68  
**Weighted Average:** Precision 0.80, Recall 0.80, F1-Score 0.80

---

## Key Analysis Points

- **Data Pre-processing**
  Best implementation of the pre-processing function so far. 
  See `data\cleaned_alerts.csv` for more information.

- **Data Split:**  
  The model was trained with a 70/30 split, providing a larger training set (70%) while evaluating on a 30% test set.

- **Priority Performance:**  
  - Overall Priority accuracy improved to 84% (from 81% in Version 2).  
  - Class 2 shows a notable improvement with high precision, recall, and F1-scores.
  - Underrepresented Class 3 remains challenging, with lower scores.

- **Taxonomy Performance:**  
  - Overall Taxonomy accuracy remains at 80%.  
  - Most classes perform well; however, minority classes (e.g., Class 0 and Class 9) still exhibit lower performance.

- **Improvements vs. Version 2:**  
  - **Priority:** The overall accuracy increased from 81% to 84%, driven primarily by improved performance in Class 2.
  - **Taxonomy:** Results are very similar, suggesting that further improvements in feature representation or addressing class imbalance are needed for minority classes.

---

## Conclusion

Version 3’s improved preprocessing appears to have slightly boosted Priority prediction accuracy while Taxonomy prediction remains consistent. The changes in cleaning the description text helped reduce noise, especially benefiting classes with clearer textual cues. Future work should focus on addressing class imbalance and possibly exploring enhanced text vectorization techniques to further improve predictions, particularly for underrepresented classes.
