import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"C:\machine_learning\Day22\loan\loan_train.csv")

# Handle missing values
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# Encode
df['Self_Employed'] = df['Self_Employed'].map({'No': 0, 'Yes': 1})
df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1})

# Features
X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Self_Employed']]
y = df['Loan_Status']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Models (IMPORTANT: probability=True)
svm_linear = SVC(kernel='linear', probability=True)
svm_poly = SVC(kernel='poly', degree=3, probability=True)
svm_rbf = SVC(kernel='rbf', probability=True)

# Train
svm_linear.fit(X_train, y_train)
svm_poly.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)

# Save models
pickle.dump(svm_linear, open(r"C:\machine_learning\Day22\loan\svm_linear.pkl", "wb"))
pickle.dump(svm_poly, open(r"C:\machine_learning\Day22\loan\svm_polynomial.pkl", "wb"))
pickle.dump(svm_rbf, open(r"C:\machine_learning\Day22\loan\svm_rbf.pkl", "wb"))
pickle.dump(scaler, open(r"C:\machine_learning\Day22\loan\scaler.pkl", "wb"))

print("✅ Models saved successfully")
