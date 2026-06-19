import pandas as pd
import numpy as np

# ─── 1. GENERATE SYNTHETIC DATA ───────────────────────────────────────────
np.random.seed(42) # Forces the random numbers to be the same every time
N_CUSTOMERS = 500
N_TRANSACTIONS = 5000

# Create realistic dates over the last 2 years
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')

df = pd.DataFrame({
    'customer_id':    np.random.choice([f'C{i:04d}' for i in range(1, N_CUSTOMERS+1)], N_TRANSACTIONS),
    'transaction_id': [f'T{i:05d}' for i in range(1, N_TRANSACTIONS+1)],
    'purchase_date':  pd.to_datetime(np.random.choice(dates, N_TRANSACTIONS)),
    'amount':         np.round(np.random.exponential(scale=100, size=N_TRANSACTIONS), 2)
})

# Drop negative or zero amounts
df = df[df['amount'] > 0]

print(f"Raw Data: {len(df)} transactions.")


# ─── 2. CLEAN AND SCRUB THE DATA ──────────────────────────────────────────

# Fix 1: Drop Duplicates
df = df.drop_duplicates(subset='transaction_id')

# Fix 2: Cap the Outliers
cap = df['amount'].quantile(0.99)
df['amount'] = df['amount'].clip(upper=cap)

print(f"Cleaned Data: {len(df)} transactions.")
print(f"Maximum purchase amount capped at: ${cap:.2f}")


# ─── 3. CALCULATE RAW RFM ─────────────────────────────────────────────────

# We pin a reference date so our "days ago" math doesn't change every day
REFERENCE_DATE = pd.Timestamp('2024-01-01')

# Collapse all transactions into one row per customer
rfm = df.groupby('customer_id').agg(
    last_purchase=('purchase_date', 'max'),
    frequency=('transaction_id', 'count'),
    monetary=('amount', 'sum')
).reset_index()

# Calculate Recency: How many days between their last purchase and our reference date?
rfm['recency'] = (REFERENCE_DATE - rfm['last_purchase']).dt.days
rfm = rfm.drop(columns='last_purchase')


# ─── 4. SCORE RFM (1-5 SCALE) ─────────────────────────────────────────────
# We use pd.qcut to divide customers into 5 equal-sized buckets (quintiles).

# Recency is INVERTED: A lower number of days is BETTER, so they get a 5.
rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)

# Frequency & Monetary: Higher is BETTER.
# We use rank(method='first') on frequency to forcefully break ties.
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm['m_score'] = pd.qcut(rfm['monetary'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)

# Combine them into a string identifier (e.g., "555" for the best, "111" for the worst)
rfm['rfm_label'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)

print("\n--- Customer Profiles (First 5) ---")
print(rfm[['customer_id', 'recency', 'frequency', 'monetary', 'rfm_label']].head())
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ─── 5. RULE-BASED SEGMENTATION ───────────────────────────────────────────
# We define the business logic to translate 1-5 scores into human words.
def assign_segment(row):
    r, f, m = row['r_score'], row['f_score'], row['m_score']
    
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif f >= 4 and m >= 4:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'Potential Loyalists'
    elif r >= 3 and f >= 3:
        return 'Promising'
    elif r <= 2 and f >= 4:
        return 'At Risk'
    elif r <= 2 and f >= 2:
        return 'Need Attention'
    elif r == 1 and f == 1:
        return 'Lost'
    else:
        return 'Others'

# Apply the rules to every single row
rfm['segment'] = rfm.apply(assign_segment, axis=1)

print("\n--- Segment Counts (Human Rules) ---")
print(rfm['segment'].value_counts())

# ─── 6. MACHINE LEARNING (K-MEANS) ────────────────────────────────────────
# K-Means doesn't care about our 1-5 scores. It wants the raw, unadulterated numbers.
features = rfm[['recency', 'frequency', 'monetary']].copy()

# CRITICAL STEP: Scaling
# K-Means calculates literal geometric distance between data points. 
# A difference of $1,000 in monetary value will mathematically obliterate 
# a difference of 5 purchases in frequency. StandardScaler forces all 
# three variables onto a level playing field.
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# We tell the algorithm to find exactly 4 distinct tribes of customers.
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['cluster'] = kmeans.fit_predict(features_scaled)
rfm['cluster'] = rfm['cluster'].apply(lambda x: f'Cluster {x+1}')

print("\n--- Machine Learning Clusters ---")
print(rfm['cluster'].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── 7. VISUALIZATION DASHBOARD ───────────────────────────────────────────
# Create a folder to hold the charts so they don't clutter your directory
os.makedirs('outputs/figures', exist_ok=True)

# Set the visual style
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (10, 6)

print("\nGenerating charts... close each chart window to generate the next one.")

# Chart 1: Scatter Plot (Recency vs Monetary)
# This proves visually that our segments are distinct
plt.figure(figsize=(12, 7))
segments = rfm['segment'].unique()
palette = sns.color_palette('husl', len(segments))

for seg, color in zip(segments, palette):
    mask = rfm['segment'] == seg
    plt.scatter(rfm[mask]['recency'], rfm[mask]['monetary'], label=seg, color=color, alpha=0.7, s=50)

plt.xlabel('Recency (days since last purchase)')
plt.ylabel('Total Monetary Value ($)')
plt.title('Customer Map: Recency vs Monetary Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('outputs/figures/recency_vs_monetary.png')
plt.show()

# Chart 2: The Heatmap (The Business View)
# Shows the average 1-5 score for each segment at a glance
heatmap_data = rfm.groupby('segment')[['r_score','f_score','m_score']].mean()
heatmap_data.columns = ['Recency Score', 'Frequency Score', 'Monetary Score']

plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5)
plt.title('Average RFM Scores by Segment')
plt.tight_layout()
plt.savefig('outputs/figures/rfm_heatmap.png')
plt.show()

# ─── 8. EXPORT FINAL DATA ─────────────────────────────────────────────────
# We only export the columns the marketing team actually cares about.
output_cols = ['customer_id', 'recency', 'frequency', 'monetary', 
               'rfm_label', 'segment', 'cluster']

rfm[output_cols].to_csv('outputs/rfm_results.csv', index=False)
print("Pipeline Complete. Final data exported to 'outputs/rfm_results.csv'")