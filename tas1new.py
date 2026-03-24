import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns


# 1. Load datasets

tickets = pd.read_csv("gold_match_tickets.csv")
context = pd.read_csv("gold_match_context.csv")

tickets.columns = tickets.columns.str.strip()
context.columns = context.columns.str.strip()


# 2. Select relevant features

tickets_df = tickets[
    [
        "match_id",
        "tickets_sold_total",
        "seasonpass_holders",
        "tickets_sold_b2c",
        "tickets_sold_b2b"
    ]
]

context_df = context[
    [
        "match_id",
        "promo_tickets_total",
        "pct_free_tickets",
        "has_promotion",
        "promotion_names"
    ]
]


# 3. Merge datasets

df = pd.merge(tickets_df, context_df, on="match_id", how="inner")


# 4. Data cleaning

df["has_promotion"] = df["has_promotion"].astype(str).str.lower().map({
    "true": 1,
    "false": 0,
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0
})

# Fill missing values only if needed
df["has_promotion"] = df["has_promotion"].fillna(0)

# Drop rows with missing important values
df = df.dropna()


# 5. Feature engineering

df["paid_tickets"] = df["tickets_sold_b2c"] + df["tickets_sold_b2b"]
df["promo_ratio"] = df["promo_tickets_total"] / df["tickets_sold_total"]
df["seasonpass_ratio"] = df["seasonpass_holders"] / df["tickets_sold_total"]


df["seasonpass_pct"] = df["seasonpass_holders"] / df["tickets_sold_total"]
df["b2c_pct"] = df["tickets_sold_b2c"] / df["tickets_sold_total"]
df["b2b_pct"] = df["tickets_sold_b2b"] / df["tickets_sold_total"]


# 6. Analysis

print("\n================ SUMMARY STATISTICS ================")
print(df.describe())

print("\n================ CORRELATION WITH TARGET ================")
target_corr = df.corr(numeric_only=True)["tickets_sold_total"].sort_values(ascending=False)
print(target_corr)

print("\n================ FULL CORRELATION MATRIX ================")
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)


# 7. Promotion impact analysis

print("\n================ PROMOTION IMPACT ================")
promotion_effect = df.groupby("has_promotion")[
    ["tickets_sold_total", "promo_tickets_total", "pct_free_tickets"]
].mean()
print(promotion_effect)

print("\n================ PROMOTION DISTRIBUTION ================")
promo_sales = df.groupby("has_promotion")["tickets_sold_total"].describe()
print(promo_sales)


# 8. Promotion name analysis

print("\n================ PROMOTION TYPE ANALYSIS ================")
promotion_performance = df.groupby("promotion_names")["tickets_sold_total"].mean().sort_values(ascending=False)
print(promotion_performance)


# 9. Ticket sales composition

print("\n================ AVERAGE TICKET COMPOSITION ================")
composition = df[["seasonpass_pct", "b2c_pct", "b2b_pct"]].mean()
print(composition)


# 10. Demand segmentation

df["demand_level"] = pd.qcut(df["tickets_sold_total"], q=3, labels=["Low", "Medium", "High"])

print("\n================ DEMAND SEGMENTATION ================")
demand_analysis = df.groupby("demand_level")[
    ["seasonpass_holders", "tickets_sold_b2c", "tickets_sold_b2b", "promo_tickets_total"]
].mean()
print(demand_analysis)


# 11. Outlier detection

q1 = df["tickets_sold_total"].quantile(0.25)
q3 = df["tickets_sold_total"].quantile(0.75)
iqr = q3 - q1

outliers = df[
    (df["tickets_sold_total"] < q1 - 1.5 * iqr) |
    (df["tickets_sold_total"] > q3 + 1.5 * iqr)
]

print("\n================ OUTLIER MATCHES ================")
if len(outliers) == 0:
    print("No strong outliers found.")
else:
    print(outliers)


# 12. Regression model

features = [
    "seasonpass_holders",
    "tickets_sold_b2c",
    "tickets_sold_b2b",
    "promo_tickets_total",
    "pct_free_tickets",
    "has_promotion"
]

X = df[features]
y = df["tickets_sold_total"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n================ REGRESSION RESULTS ================")
print("Model R² score:", r2_score(y_test, predictions))

coefficients = pd.DataFrame({
    "Feature": features,
    "Impact_on_ticket_sales": model.coef_
})

print("\nFeature importance:")
print(coefficients.sort_values(by="Impact_on_ticket_sales", ascending=False))


# 13. Visualizations


# Scatter plot: promo tickets vs total sales
plt.figure(figsize=(8, 5))
plt.scatter(df["promo_tickets_total"], df["tickets_sold_total"])
plt.xlabel("Promo Tickets Total")
plt.ylabel("Tickets Sold Total")
plt.title("Promotion Tickets vs Total Ticket Sales")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Boxplot: promotion vs sales
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["has_promotion"], y=df["tickets_sold_total"])
plt.xlabel("Has Promotion")
plt.ylabel("Tickets Sold Total")
plt.title("Ticket Sales Distribution With vs Without Promotion")
plt.show()

# Bar chart: average ticket composition
plt.figure(figsize=(8, 5))
composition.plot(kind="bar")
plt.ylabel("Average Share of Total Tickets")
plt.title("Average Ticket Sales Composition")
plt.show()

# Bar chart: promotion performance
plt.figure(figsize=(10, 5))
promotion_performance.plot(kind="bar")
plt.ylabel("Average Tickets Sold")
plt.title("Average Ticket Sales by Promotion Type")
plt.show()

# Residual plot
residuals = y_test - predictions
plt.figure(figsize=(8, 5))
plt.scatter(predictions, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Ticket Sales")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.show()