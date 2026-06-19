# Customer Segmentation: RFM Analysis & K-Means Clustering

A complete data pipeline that processes e-commerce transactions to categorize customers based on their purchasing behavior. It uses traditional rule-based RFM (Recency, Frequency, Monetary) scoring and validates the business logic using K-Means machine learning clustering.

## What It Does
1. **Data Generation:** Synthesizes 5,000 realistic e-commerce transactions.
2. **Data Scrubbing:** Removes duplicates and caps spending outliers at the 99th percentile to prevent cluster distortion.
3. **RFM Scoring:** Aggregates transactions and assigns a 1-5 score for Recency, Frequency, and Monetary values using quintile discretization (`pd.qcut`).
4. **Rule-Based Segmentation:** Translates numerical scores into actionable business segments (e.g., "Champions", "At Risk", "Lost").
5. **Machine Learning Validation:** Uses `StandardScaler` and `KMeans` to independently cluster the raw data and verify the human-defined segments.
6. **Data Visualization:** Generates a scatter plot and a heatmap to provide executive-level business intelligence.

## How to Run Locally (macOS/Linux)
**1. Clone the repository and navigate into it:**

git clone [https://github.com/YOUR_USERNAME/rfm-segmentation.git](https://github.com/YOUR_USERNAME/rfm-segmentation.git)
cd rfm-segmentation

**2. Create and activate a virtual environment:**

python3 -m venv rfm_env
source rfm_env/bin/activate

**3. Install dependencies:**

pip install pandas numpy scikit-learn matplotlib seaborn

**4. Execute the pipeline:**

python rfm_engine.py
**Note:** The script will generate and display charts. You must close the active chart window for the pipeline to continue to the next step.
