#### SER594: Machine Learning Evaluation
#### US Agricultural Import and Export Analysis
#### Devanshi Prajapati
#### 11/25/2024

## Evaluation Metrics for Dependency Model
### Metric 1
**Name:** Accuracy

**Choice Justification:** Accuracy is an appropriate metric for this classification problem as it provides a straightforward measure of the model's overall correctness. It's useful when the classes are balanced and all prediction errors are equally important.

**Interpretation:** The accuracy scores for the dependency models range from 0.485507 to 0.962963. Dependency1 shows the highest accuracy at 96.30%, indicating that it correctly classifies 96.30% of all instances. Dependency4 has the lowest accuracy at 48.55%, suggesting it performs only slightly better than random guessing for a binary classification task.

### Metric 2
**Name:** F1 Score

**Choice Justification:** The F1 score is chosen because it provides a balanced measure of precision and recall. It's  useful when there is an uneven class distribution and when false positives and false negatives have different costs.


**Interpretation:** The F1 scores range from 0.653659 to 0.981132. Dependency1 has the highest F1 score of 0.9811, indicating a very good balance between precision and recall. Dependency4 has the lowest F1 score of 0.6537, suggesting a moderate performance in balancing precision and recall.

## Evaluation Metrics for Dependency Model
### Metric 1
**Name:** RMSE

**Choice Justification:** Root Mean Square Error (RMSE) is appropriate for this regression problem as it measures the standard deviation of the prediction errors. It's useful when large errors are particularly undesirable, as it gives higher weight to large errors.

**Interpretation:**  The RMSE values for the seasonal models range from 2.954860e+08 to 3.071955e+08. These values are quite large, indicating substantial prediction errors. Seasonal1 has the lowest RMSE, suggesting it performs slightly better than the other models, but the difference is minimal.

### Metric 2
**Name:** R-Square

**Choice Justification:** R-Squared is chosen because it represents the proportion of variance in the dependent variable that is predictable from the independent variables. It provides an easy-to-understand measure of how well the model fits the data.

**Interpretation:** All R-squared values are negative, ranging from -0.003578 to -0.084694. Negative R-squared values indicate that the models perform worse than a horizontal line (mean of the data). This suggests that all seasonal models are performing poorly and are not explaining any of the variability in the target variable.

## Alternative Models for Dependency
### Initial Model 
**Construction:** The initial model is dependency1, which uses a RandomForestClassifier with minimal complexity. It includes 1 estimator, a max depth of 2, and larger leaf sizes to ensure generalization.

**Evaluation:** This model achieved an accuracy of 0.962963 and an F1 score of 0.981132, indicating excellent performance with simple patterns and balanced precision-recall trade-offs.

### Alternative 1
**Construction:** Dependency2 increases the number of trees to 3 and the max depth to 3, aiming for slightly more robustness while maintaining simplicity.

**Evaluation:** With an accuracy of 0.870370 and an F1 score of 0.930693, this model shows good performance but slightly less effective than dependency1, suggesting that the added complexity did not significantly improve results.

### Alternative 2
**Construction:** Dependency3 further increases the number of trees to 5 and max depth to 4, allowing for more complex patterns to be captured.

**Evaluation:** This model achieved an accuracy of 0.833333 and an F1 score of 0.909091. The performance is moderate, indicating diminishing returns from increased complexity compared to simpler models.

### Alternative 3
**Construction:** Dependency4 uses 10 trees with a max depth of 6, allowing deeper splits for capturing intricate patterns.

**Evaluation:** The model's accuracy is 0.485507 with an F1 score of 0.653659, indicating poor performance likely due to overfitting or inappropriate complexity for the data.

## Best Model for Dependency

**Model:** Dependency1 is the best model due to its highest accuracy and F1 score, demonstrating effective performance with minimal complexity.

## Alternative Models for Seaonality
### Initial Model 
**Construction:** The initial model is seasonal1, using an MLPRegressor with a single hidden layer of 50 neurons and ReLU activation.

**Evaluation:** This model has an RMSE of approximately 2.954860e+08 and an R-squared of -0.003578, indicating poor fit and high error rates.

### Alternative 1
**Construction:** Seasonal2 introduces two hidden layers (64 and 32 neurons) with tanh activation, aiming for better non-linearity capture.

**Evaluation:** This model has an RMSE of approximately 3.028833e+08 and an R-squared of -0.054455, showing no improvement in fit over seasonal1.

### Alternative 2
**Construction:** Seasonal3 uses three hidden layers (100, 50, and 25 neurons) with tanh activation and early stopping to prevent overfitting.

**Evaluation:** The RMSE is approximately 3.028835e+08 with an R-squared of -0.054457, similar to seasonal2, indicating persistent poor performance.

### Alternative 3
**Construction:** Seasonal4 features three hidden layers (128, 64, and 32 neurons) with ReLU activation for potentially better performance on complex data.

**Evaluation:** This model has an RMSE of approximately 3.071955e+08 and an R-squared of -0.084694, showing the worst fit among all models tested.

## Best Model for Seasonality

**Model:** None of the models perform well based on R-squared values, suggesting a need for reevaluation or alternative approaches such as feature engineering or different algorithms.
