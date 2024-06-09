import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

# Load the dataset
test = pd.read_csv('test.csv')
data = pd.read_csv('train.csv')
data = data.sample(frac=0.3, random_state=42)


# Drop the 'id' column as it is not relevant for the prediction
data = data.drop('id', axis=1)
test = test.drop('id', axis=1)

# Encode categorical variables
categorical_cols = ['Gender', 'Customer Type','Age','Type of Travel','Class','Flight Distance','Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Inflight service','Cleanliness','Departure Delay in Minutes','Arrival Delay in Minutes']
encoder = LabelEncoder()
for col in categorical_cols:
    if col in data.columns:
        data[col] = encoder.fit_transform(data[col])
        test[col] = encoder.fit_transform(test[col])
data['satisfaction'] = encoder.fit_transform(data['satisfaction'])
test['satisfaction'] = encoder.fit_transform(test['satisfaction'])

# Split the dataset into features (x_train) and target variable (y)
y_train = data['satisfaction']
x_train = data.drop('satisfaction', axis=1)
y_test = test['satisfaction']
x_test = test.drop('satisfaction', axis=1)

# Scale numerical variables
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Feature selection
n_features_to_select = 10  # change this number to select a different number of features
selector = RFE(estimator=RandomForestClassifier(n_jobs=-1), n_features_to_select=n_features_to_select)
selector.fit(x_train, y_train)
x_train_selected = selector.transform(x_train)
x_test_selected = selector.transform(x_test)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1), param_grid=param_grid, cv=5)
grid_search.fit(x_train_selected, y_train)
print(grid_search.best_params_)

# Ensemble methods
bagging_clf = BaggingClassifier(estimator=RandomForestClassifier(n_jobs=-1), n_estimators=10, random_state=42)
bagging_clf.fit(x_train_selected, y_train)

# Cross-validation
scores = cross_val_score(estimator=RandomForestClassifier(), X=x_train_selected, y=y_train, cv=5)
print(scores.mean())

# Handling class imbalance
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_selected, y_train)

# Train the RandomForestClassifier model
model = RandomForestClassifier(n_jobs=-1)
model.fit(x_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = model.predict(x_test_selected)

# Evaluate the model
print(classification_report(y_test, y_pred))
