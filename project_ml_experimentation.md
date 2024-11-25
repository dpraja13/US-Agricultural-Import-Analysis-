#### SER594: Experimentation
#### US Agricultural Import and Export Analysis
#### Devanshi Prajapati
#### 11/25/2024


## Explainable Records for Dependency
### Record 1
**Raw Data:** State: AK, Fiscal year: 2023, Fiscal quarter: 2, Imports: 1905716.0, Exports: 1618574.0, Ratio: 1.1774043077424943, Dependency Level: Medium, Encoded Label: 2, Predicted Label: Low

**Prediction Explanation:** The model predicts a Low dependency level for Alaska in Q2 2023, which differs from the actual Medium level. This prediction might be reasonable because:
- The import-export ratio is close to 1 (1.177), indicating a near balance between imports and exports.
- Alaska's economy is known for its natural resources, particularly oil and fishing, which could lead to significant exports.
- The absolute values of imports and exports are relatively low, which might suggest less overall trade dependency.
While the model's prediction doesn't match the actual label, it can be argued that the low absolute trade values and near-balanced ratio justify a low dependency classification.

### Record 2
**Raw Data:** State: UT, Fiscal year: 2019, Fiscal quarter: 4, Imports: 52486670.0, Exports: 91767554.0, Ratio: 0.5719523700065058, Dependency Level: Low, Encoded Label: 1, Predicted Label: Low

**Prediction Explanation:** The model correctly predicts a Low dependency level for Utah in Q4 2019. This prediction is reasonable because:
- The import-export ratio is 0.572, indicating that exports significantly exceed imports.
- Utah's exports are nearly double its imports, suggesting a strong export-oriented economy.
- A low dependency level aligns with an economy that exports more than it imports, as it's less reliant on external goods.
The model's prediction matches the actual label and is supported by the economic principle that higher exports relative to imports often indicate lower trade dependency.

## Explainable Records for Seasonality
### Record 1
**Raw Data:** State: AL, Commodity name: Pet Food, Fiscal year: 2024, Fiscal quarter: 2, Season: Spring, Dollar value: 2180102, Predicted Dollar Value: 51142674.21129462

**Prediction Explanation:** The model's prediction for Alabama's pet food trade in Spring 2024 is significantly higher than the actual value. While this large discrepancy might seem unreasonable, there could be domain-specific explanations:
- Seasonal trends: Spring might typically see a surge in pet food demand due to increased outdoor activities and pet adoptions.
- Economic projections: The model might be accounting for projected growth in the pet food industry, which has been expanding rapidly in recent years.
- Regional factors: Alabama might be expected to become a major hub for pet food production or distribution, leading to higher predicted values.
However, the magnitude of the difference suggests that the model may be overfitting or failing to capture important nuances in the data.

### Record 2
**Raw Data:** OH, Commodity name: Industrial Alcohols And Fatty Acids, Fiscal year: 2023, Fiscal quarter: 3, Season: Summer, Dollar value: 32010647, Predicted Dollar Value: 51166292.37354974

**Prediction Explanation:** The model's prediction for Ohio's industrial alcohols and fatty acids trade in Summer 2023 is higher than the actual value, but the discrepancy is less extreme than in the previous example. This prediction could be considered more reasonable because:
- Seasonal impact: Summer might typically see increased production and trade of industrial alcohols and fatty acids due to higher demand in related industries.
- Economic factors: Ohio has a significant manufacturing sector, and the model might be accounting for expected growth in industries that use these chemicals.
- Market trends: There could be an anticipated increase in demand for these products due to their use in sanitizers, cleaning products, or industrial processes.
While the prediction is still notably higher than the actual value, it's within a more plausible range and could reflect potential growth or seasonal peaks in this specific commodity trade.

## Interesting Features
### Feature A
**Feature:** State

**Justification:** The "State" feature is relevant to both the classification and seasonality predictions because it encapsulates regional economic characteristics, policies, and industrial strengths that influence trade patterns. Different states have varying levels of industrialization, natural resources, and economic policies that affect their import-export activities. For classification, the state can impact the dependency level due to its unique economic structure, while for seasonality, it affects how commodities are traded throughout the year based on local demand and supply conditions.

### Feature B
**Feature:** Fiscal Quarter

**Justification:** The "Fiscal Quarter" feature is significant for both predictions as it captures temporal variations in trade activities. For classification, different quarters might reflect changes in import-export ratios due to seasonal demand or supply shifts, impacting dependency levels. In terms of seasonality, the fiscal quarter directly relates to seasonal trends and fluctuations in commodity trade, as certain products may have higher demand or production in specific quarters due to climatic or economic cycles.


## Experiments for Dependency Predictions
### Varying A: Ratio
**Prediction Trend Seen:** As the Ratio (imports to exports) increases, the model tends to predict higher dependency levels. For example, when the Ratio is low (e.g., 0.5719 for Utah in Q4 2019), the model predicts "Low" dependency. As the Ratio increases (e.g., 1.1774 for Alaska in Q2 2023), the model still predicts "Low", but for higher ratios (e.g., 4.7485 for Alaska in Q1 2022), it predicts "High" dependency. This suggests a positive correlation between the Ratio and predicted dependency level.

### Varying B: Fiscal Quater
**Prediction Trend Seen:** Varying the Fiscal quarter alone doesn't show a clear trend in dependency predictions. For instance, Alaska's predictions for different quarters in 2023 are all "Low" despite varying quarters. This suggests that the model doesn't heavily weight the Fiscal quarter in isolation for determining dependency levels.

### Varying A and B together
**Prediction Trend Seen:** When varying both Ratio and Fiscal quarter together, the Ratio appears to dominate the prediction trend. For example, Alaska's Q2 2023 with a Ratio of 1.1774 and Q3 2023 with a Ratio of 16.7423 both have different quarters but very different Ratios, resulting in "Low" and "High" predictions respectively. This indicates that the combined effect is primarily driven by changes in the Ratio rather than the Fiscal quarter.

### Varying A and B inversely
**Prediction Trend Seen:** When varying Ratio and Fiscal quarter inversely, the trend still appears to be dominated by the Ratio. For instance, comparing Alaska's Q1 2024 (Ratio 3.3790, predicted "High") with Q2 2024 (Ratio 0.6432, predicted "Low"), we see that despite the inverse change in quarter, the prediction follows the Ratio change. This further confirms that the Ratio has a stronger influence on the model's predictions than the Fiscal quarter.

## Experiments for Seasonality based Dollar Predictions
### Varying A: Fiscal Quater
**Prediction Trend Seen:** When varying the Fiscal quarter, there isn't a clear trend in the predicted dollar values. For example, Alabama's Pet Food predictions for Q1 2024 ($51,117,489) and Q2 2024 ($51,142,674) are very similar despite different quarters. This suggests that the model doesn't heavily weight the Fiscal quarter for dollar value predictions.

### Varying B: Commodity Name
**Prediction Trend Seen:** Varying the Commodity name shows some differences in predicted values, but not consistently large ones. For instance, Alabama's Q2 2024 predictions for Pet Food ($51,142,674) and Industrial Alcohols And Fatty Acids ($51,166,292) are similar. This indicates that the Commodity name alone doesn't dramatically impact predictions.

### Varying A and B together
**Prediction Trend Seen:** When varying both Fiscal quarter and Commodity name together, we see some variations in predictions, but they're not drastically different. For example, comparing Alabama's Pet Food in Q2 2024 ($51,142,674) with Ohio's Industrial Alcohols And Fatty Acids in Q3 2023 ($51,166,292), we see a small difference. This suggests that the combined effect of these features does influence predictions, but not dramatically.

### Varying A and B inversely
**Prediction Trend Seen:** When varying Fiscal quarter and Commodity name inversely, the trend is similar to varying them together. The predictions show some variations, but they're relatively small. For instance, comparing Alabama's Pet Food in Q2 2024 ($51,142,674) with Alabama's Industrial Alcohols And Fatty Acids in Q3 2023 ($51,156,962), we see only minor differences. This further confirms that while these features do impact predictions, their influence is not very strong in isolation.

