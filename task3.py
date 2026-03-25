#step 1: Libariries import:
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#step 2 loading data from csv files
match = pd.read_csv("gold_match.csv")
tickets = pd.read_csv("gold_match_tickets.csv")

#step 3: combine the datasets
df = match.merge(tickets, on="match_id")

#step 4: We want only to see matches that played at home
df = df[df["is_home_match"] == True]

#step 5: Target
y = df["tickets_scanned"]

#step 6: choosing features 
features = [
    "away_team",
    "stage",
    "matchday",
    "last_result_vs_opponent",
    "competition_name",
    "season"
]

#step 7: convert categorical data
#one hot encoding
X = pd.get_dummies(df[features])

#step 8: train test spit
#80% train 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#step 9:
#train the model with linair regression
model = LinearRegression()

model.fit(X_train, y_train)

#step 10:
#  make the prediciton
predictions = model.predict(X_test)

# step 11: evoluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)


# example how Mean Absolute errors work
# (500 + 500 + 400) / 3 = 466
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

#step 12: Looking at feature importance
#  cofficeint tels us how many factors influence the attendance
#Because Main quastion: Which factors actually matter?

importance = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_
})

importance = importance.sort_values(by="coefficient", ascending=False)

print(importance.head(15))

#step 13: show most important oppenent
#show avarage attendance per oppenet

attendance_opponent = df.groupby("away_team")["tickets_scanned"].mean()

print(attendance_opponent.sort_values(ascending=False))

#step 14: Show top 10 matches
top_matches = df.sort_values(by="tickets_scanned", ascending=False)


# graphic for visuol proof
import matplotlib.pyplot as plt

# Average attendance per opponent
attendance_opponent.sort_values(ascending=False).plot(kind='bar', figsize=(12,6))
plt.ylabel("Average Attendance")
plt.title("Average attendance per opponent")
plt.show()


#The second chart → “Model: how much influence a particular opponent or the importance
#  of the match has on attendance, after accounting for all other factors.”
importance_top = importance[importance['feature'].str.contains('away_team|stage')]
importance_top.plot(kind='bar', x='feature', y='coefficient', figsize=(12,6))
plt.title("Impact off opponent and the match importance")
plt.show()

print(top_matches[["match_date","away_team","tickets_scanned"]].head(10))

# Conclusion Question: How much do the opponent and match importance drive fans through the gate?  
#Answer: The opponent and the stage of the competition are important factors. Popular opponents such
#  as Anderlecht and Club Brugge attract an average of 9–10k fans, while less popular teams draw
#  only about 4–6k. Match importance and recent results add an extra effect, 
# bringing in roughly 1–2k more fans for important or successful games.

