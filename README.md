# 📊 Customer Churn Prediction

## 🧠 Overview

**Customer Churn Prediction** is a Machine Learning project that predicts whether a customer is likely to leave a company (churn) based on their past interactions and service usage data.  
This project helps businesses take **data-driven retention actions** by identifying at-risk customers early.

---

## 🚀 Objective

To build and evaluate multiple ML models that predict customer churn using demographic, billing, and service data — ultimately enabling companies to reduce customer attrition rates.

---

## 🗂️ Dataset

**Source:** [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)

**Description:**  
The dataset contains information about telecom customers, including their demographic details, account information, and usage patterns.

| Column | Description |
|--------|--------------|
| `customerID` | Unique ID for each customer |
| `gender` | Male or Female |
| `SeniorCitizen` | Whether the customer is a senior citizen |
| `Partner`, `Dependents` | Family details |
| `tenure` | Number of months the customer has stayed |
| `PhoneService`, `InternetService`, etc. | Services subscribed |
| `Contract`, `PaymentMethod` | Billing details |
| `MonthlyCharges`, `TotalCharges` | Financial features |
| `Churn` | Target variable (Yes = churned, No = retained) |

---

## ⚙️ Project Workflow

### **1️⃣ Data Preprocessing**
- Handled missing and incorrect values (`TotalCharges` column had blanks).
- Converted categorical variables using **Label Encoding** and **One-Hot Encoding**.
- Scaled numerical features using **StandardScaler**.
- Balanced the dataset using **SMOTE** (to fix class imbalance).

### **2️⃣ Exploratory Data Analysis (EDA)**
- Visualized churn distribution and correlations using **Matplotlib** and **Seaborn**.
- Identified key factors affecting churn:
  - Month-to-month contracts
  - Higher monthly charges
  - Shorter tenure

### **3️⃣ Model Building**
Implemented multiple algorithms:
- Logistic Regression  
- Random Forest  
- XGBoost  

Trained models on **80% training** and **20% testing** split.

### **4️⃣ Model Evaluation**
Metrics used:
- Accuracy  
- Precision, Recall, F1-Score  
- ROC-AUC Score  
- Confusion Matrix  

Best performance achieved by:
> ✅ **Random Forest Classifier** — Accuracy: ~85%, ROC-AUC: ~0.88

### **5️⃣ Feature Importance**
The top features driving churn were:
- `Contract`
- `MonthlyCharges`
- `tenure`
- `OnlineSecurity`
- `TechSupport`

---

## 🧮 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| Programming | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | Scikit-learn, XGBoost |
| Data Balancing | imbalanced-learn (SMOTE) |
| Environment | Jupyter Notebook / VS Code |

---

## ⚙️ Installation & Setup
```bash
git clone https://github.com/yourusername/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction

Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

Install dependencies
pip install -r requirements.txt


Run the notebook
notebooks/churn_prediction.ipynb


📈 Results
Model	Accuracy	ROC-AUC	F1-Score
Logistic Regression	0.80	0.85	0.78
Random Forest	0.85	0.88	0.84
XGBoost	0.84	0.87	0.83

✅ Best Model: Random Forest — balanced precision and recall with strong interpretability.

🧭 Business Insights

Customers on month-to-month contracts are 3x more likely to churn.

Customers with high monthly charges are more likely to leave.

Long-term contracts and bundled services increase retention.

🌐 Future Improvements

Hyperparameter tuning with GridSearchCV / RandomizedSearchCV

Deploying the model via Flask / Streamlit Web App

Building a Tableau / Power BI dashboard for churn visualization

Integration with a CRM system for real-time alerts

📢 Author

👤 Ashutosh Kushwaha
