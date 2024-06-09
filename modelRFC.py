# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
test = pd.read_csv('test.csv')
data = pd.read_csv('train.csv')

# Drop the 'id' column as it is not relevant for the prediction
data = data.drop('id', axis=1)
test = test.drop('id', axis=1)


# Encode categorical variables
categorical_cols = ['Gender', 'Customer Type','Age','Type of Travel','Class','Flight Distance','Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Inflight service','Cleanliness','Departure Delay in Minutes','Arrival Delay in Minutes']

encoder = LabelEncoder()
for col in categorical_cols:
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


# Train the XGBoost model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
