import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, accuracy_score 
# Load the dataset 
file_path = r"C:\Users\Shreyash Musmade\Desktop\Practical\AICS\AICS_Prac-2\TCP-SYNC DATASET.csv"    
df = pd.read_csv(file_path) 
# Drop non-numeric columns except the target label 
df_cleaned = df.drop(columns=["Flow ID", "Src IP", "Dst IP", "Timestamp"]).dropna() 
# Encode the target variable 
label_encoder = LabelEncoder() 
df_cleaned["Label"] = label_encoder.fit_transform(df_cleaned["Label"]) 
# Split features and target 
X = df_cleaned.drop(columns=["Label"]) 
y = df_cleaned["Label"] 
# Standardize the features 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
# Split into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) 
# Train a Random Forest classifier 
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train, y_train) 
# Make predictions 
y_pred = model.predict(X_test) 
# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred) 
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_) 
print(f"Model Accuracy: {accuracy * 100:.2f}%") 
print("Classification Report:\n", report) 