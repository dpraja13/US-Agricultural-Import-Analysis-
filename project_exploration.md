#### SER594: Exploratory Data Munging and Visualization
#### Topic: US Agricultural Import Analysis
#### Author: Devanshi Prajapati
#### Date: 10/21/2024

## Basic Questions
**Dataset Author(s):** United States Department of Agriculture (USDA), Economic Research Service (ERS)

**Dataset Construction Date:** The datasets are updated regularly. The most recent data that it was updated on was 9/12/2024. 

**Dataset Record Count:** There are different sets of data, one corresponding to state-wise imports and the other to state-wise exports. There are 39961 records in the data for imports and 40580 records for exports. 

**Dataset Field Meanings:** The different feilds included in the datasets are - 
1. State: The U.S. state to which the trade data applies
1. Fiscal year: The U.S. government fiscal year to which the data applies (October 1 to September 30)
1. Country: The country of origin for imports or destination for exports
1. Commodity name: The specific agricultural product or category being traded
1. Dollar value: The monetary value of the trade in U.S. dollars
1. Fiscal quarter: The quarter of the fiscal year (1-4, with 0 representing the full year total)
1. Trade type: Indicates whether the data represents imports or exports
1. Fiscal quarter description: Text description of the fiscal quarter or "Fiscal Year Total"

**Dataset File Hash(es):** 
1. *URL*: "https://www.ers.usda.gov/webdocs/DataFiles/100812/Top%205%20Agricultural%20Imports%20by%20State.csv?v=1388"\
*MD5 Hash*: f2852362b6d2dd72d5f28ae4dd7dd9da\

1. *URL*: "https://www.ers.usda.gov/webdocs/DataFiles/100812/Top%205%20Agricultural%20Exports%20by%20State.csv?v=1388"\
*MD5 Hash*: f5392119b0594cbd7ea114967aedbb8b\

## Interpretable Records
### Record 1
**Raw Data:** State, Fiscal Year, Country, Commodity Name, Dollar Value, Fiscal Quater, Trade Type, Fiscal Quater Description\
AK, 2024, World, Nursery products and cut flowers, 1069000, 3, Imports, Fiscal Quarter 3

**Interpretation:** This record shows the import of nursery products and cut flowers. 
- In fiscal year 2024, during the third quarter, Alaska imported nursery products and cut flowers worth $1,069,000 from various countries (represented by "World").
- This represents imports for a specific product category going from April 2024 to June 2024 (represented by "Fiscal Quarter 3").
- The data is reasonable for Alaska, considering its climate limitations for year-round cultivation of certain plants and flowers.

### Record 2
**Raw Data:** State, Fiscal Year, Country, Commodity Name, Dollar Value, Fiscal Quater, Trade Type, Fiscal Quater Description\
AK, 2024, World, Other feeds meals and fodders, 1571792, 3, Exports,  Fiscal Quarter 3

**Interpretation:** This record shows the export of Other feeds, meals, and fodders. 
- In fiscal year 2024, during the third quarter, Alaska exported other feeds, meals, and fodders worth $1,571,792 to various countries (represented by "World").
- This represents exports for a specific product category going from April 2024 to June 2024 (represented by "Fiscal Quarter 3").
- The data is reasonable given Alaska's fishing industry, which likely produces byproducts used in animal feed.

## Background Domain Knowledge
Agricultural trade plays a key role in the global economy, affecting food security, economic growth, and international relations. The U.S., being one of the largest agricultural producers and traders, has a big impact on global markets. Understanding U.S. agricultural trade patterns at the state level is important for policymakers, economists, and industry experts.

Agricultural trade includes the buying and selling of crops, livestock, and processed foods. Trade patterns are shaped by many factors like climate, soil, technology, consumer preferences, and government policies. Each U.S. state has its own agricultural strengths, contributing to the diversity of the country’s overall agricultural trade.

The importance of agricultural trade goes beyond economics. As the 'American Enterprise Institute' (https://www.aei.org/research-products/report/why-agricultural-trade-is-or-can-be-a-life-and-death-matter/) notes, agricultural trade can be a “life and death matter,” especially for developing countries that rely on food imports to meet their needs. For the U.S., agricultural trade helps manage surplus production and supports rural economies.

Over the past century, U.S. agricultural trade has changed dramatically. According to the 'USDA' (https://www.usda.gov/media/blog/2024/02/21/100-years-agricultural-trade-century-growth-innovation-and-progress), U.S. agricultural exports grew from $1.9 billion in 1920 to $177.8 billion in 2022. This growth reflects more production, higher global demand, better transportation and storage, and changing trade policies.

Analyzing state-level agricultural trade provides useful insights into regional specializations. Some states are top exporters due to favorable conditions or strong industries, while others rely more on imports to meet demand. Knowing these patterns helps with decisions on infrastructure, policy, and agricultural research.

U.S. states are also affected by global market conditions and trade agreements. As 'Michigan State University' (https://globaledge.msu.edu/industries/agriculture/background) explains, factors like exchange rates, trade barriers, and global demand shape agricultural trade.

In summary, looking at state-level agricultural trade data gives a deeper understanding of how U.S. regions contribute to and are impacted by global markets. This knowledge is important for creating effective policies in an increasingly connected world.

## Dataset Generality
The USDA Economic Research Service dataset on state-level agricultural trade is a reliable snapshot of real-world trade patterns in the U.S. It covers all 50 states, ensuring no regional trends are missed, and spans multiple fiscal years, allowing us to track both short-term changes and long-term shifts. The dataset includes a wide range of commodities, from major crops like corn and soybeans to more niche products, reflecting the true variety of agricultural trade across the country.

With detailed data on trade partners and quarterly updates, it captures important seasonal shifts, especially for perishable goods like fruits and vegetables. The inclusion of key trade relationships, like those with Canada, China, and Mexico, highlights the U.S.’s role in global agriculture.

As an official government source, the data is accurate and regularly updated, making it perfect for analyzing trends and understanding how different states engage in agricultural trade.

## Data Transformations
### Transformation 1 - Removing Duplicates
**Description:** The code removes duplicate rows from the dataset using df = df.drop_duplicates().

**Soundness Justification:** Removing duplicates improves data quality by eliminating redundant information without changing the semantics of the data. No usable data is discarded since unique information is retained. This operation does not introduce errors or outliers.

### Transformation 2 - Handling Missing Values
**Description:** The code removes rows with missing values using df = df.dropna().

**Soundness Justification:** This ensures that the remaining data is complete and reliable. While it does discard some data, it's justified when the number of missing values is small relative to the dataset size. It does not change the semantics of the remaining data or introduce errors.

### Transformation 3 - Correcting Data Types
**Description:**  The code converts 'Fiscal year', 'Fiscal quarter', and 'Dollar value' to appropriate numeric types using pd.to_numeric().

**Soundness Justification:** This ensures data consistency and enables proper numerical operations. It does not change the semantics of the data, as it's merely converting the representation, not the values themselves. Using errors='coerce' maintains data integrity by converting non-numeric values to NaN.

### Transformation 4 - Standardizing Categorical Columns
**Description:** The code standardizes 'Country' and 'Commodity name' columns by trimming spaces and applying title case.

**Soundness Justification:** This improves data consistency without changing the fundamental meaning of the data. Trimming spaces removes irrelevant whitespace, and applying title case ensures a uniform format. This does not discard usable data, introduce errors, or create outliers.

### Transformation 5 - Removing Rows with Fiscal Quarter 0
**Description:** The code filters out rows where the 'Fiscal quarter' is 0 using df = df[df['Fiscal quarter'] != 0].

**Soundness Justification:** This removes annual summary rows, which are redundant given the presence of quarterly data. It does not change the semantics of the remaining data or introduce errors. While it does discard some data, this data is a sum of other rows, so no unique information is lost. This transformation improves the consistency of the dataset for analysis focused on quarterly trends.

## Visualizations
### Visual 1 - Scatter Plot of Dollar Value vs Fiscal Quarter
**Analysis:** The scatter plot indicates consistent dollar values across fiscal quarters for both exports and imports. This suggests stable trade flow throughout the year, with no significant seasonal spikes or drops.

### Visual 2 - Scatter Plot of Fiscal Quarter vs Fiscal Year
**Analysis:** The plot shows distinct clusters for each fiscal year, indicating regular reporting across quarters. This consistency suggests systematic data collection practices over the years, ensuring reliable trend analysis.

### Visual 3 - Scatter Plot of Dollar Value vs Fiscal Year
**Analysis:** The scatter plot demonstrates variability in dollar value across fiscal years, with some years showing higher concentrations of trade. This could reflect economic conditions or policy changes impacting trade volumes in specific years.


### Visual 4 - Histogram of Countries
**Analysis:** The distribution reveals a concentration of trade with a few countries, with the United States and China being prominent partners. This suggests strategic economic relationships and dependencies on these major trading partners.

### Visual 5 - Histogram of Commodity Names
**Analysis:** The histogram shows that certain commodities, such as "Soybeans" and "Corn," dominate both exports and imports. This indicates these commodities are key players in trade activities, reflecting their importance in agricultural markets.



