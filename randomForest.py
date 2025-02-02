import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('clear_data.csv')

categorical_features = ["city", "district", "neighbourhood"]
numerical_features = ["room", "living_room", "size", "age", "floor"]


full_pipeline = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])


X = df.drop('price', axis=1)
y = df['price']


bins = [x for x in range(900000, 10000001, 1000000)]
labels = [x for x in range(1, len(bins))]
print(bins)
print(labels)

y = pd.cut(y, bins=bins, labels=labels)

#print(y.unique())  # Kaç tane NaN var gösterir


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Pipeline([
    ('preprocessing', full_pipeline),
    ('model', RandomForestClassifier(n_estimators=100))
])


model.fit(X_train, y_train)


y_predict = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_predict)

new_df = pd.DataFrame(conf_matrix)

print(new_df)


custom_house = {
    "city": ["izmir"],
    "district": ["Torbalı"],
    "neighbourhood": ["ertuğrul"],
    "room": [2],
    "living_room": [1],
    "size": [120],  # in square meters
    "age": [0],  # in years
    "floor": [2]
}

custom_house_df = pd.DataFrame(custom_house)


print(f"Estimated price category: {model.predict(custom_house_df)[0]}")








