# 🏦 CreditWise Loan System

> An intelligent ML-powered loan approval system for SecureTrust Bank

---

## 📌 Problem Statement

**SecureTrust Bank** is a mid-sized financial company offering personal and home loans to customers across urban and rural regions of India. Every day, hundreds of customers apply for loans through online and branch applications.

### The Challenge

Until now, the bank has relied on a **manual verification process** where loan officers evaluate applications by checking:
- Income proofs
- Employment details
- Credit history
- Other documents

This process is **time-consuming, biased, and inconsistent**, leading to two major problems:

| Problem | Impact |
|---|---|
| ✅ Good customers sometimes get **rejected** | Loss of business |
| ❌ High-risk customers sometimes get **approved** | Financial losses |

### The Solution

The bank wants to introduce an **intelligent loan approval system** powered by Machine Learning that can:
- Automatically analyse applicant details
- **Predict whether a loan should be Approved or Rejected** before final human verification
- Learn hidden patterns from previous customer records
- Provide accurate, fast, and unbiased loan approval decisions

---

## 👨‍💻 Role

You are hired as a **Machine Learning Engineer** to design and develop this intelligent system using historical loan application data.

---

## 📊 Dataset Description

Each row in the dataset represents a **loan applicant** and contains multiple attributes describing their personal, financial, and credit information.

### Feature Columns

| Column | Description |
|---|---|
| `Applicant_ID` | Unique applicant ID |
| `Applicant_Income` | Monthly income of applicant |
| `Coapplicant_Income` | Monthly income of co-applicant |
| `Employment_Status` | Salaried / Self-Employed / Business |
| `Age` | Applicant age |
| `Marital_Status` | Married / Single |
| `Dependents` | Number of dependents |
| `Credit_Score` | Credit bureau score |
| `Existing_Loans` | Number of already running loans |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Savings balance |
| `Collateral_Value` | Value of collateral provided |
| `Loan_Amount` | Loan amount requested |
| `Loan_Term` | Loan duration (months) |
| `Loan_Purpose` | Home / Education / Personal / Business |
| `Property_Area` | Urban / Semi-Urban / Rural |
| `Education_Level` | Graduate / Postgraduate / Undergraduate |
| `Gender` | Male / Female |
| `Employer_Category` | Govt / Private / Self |

### 🎯 Target Column

| Column | Description |
|---|---|
| `Loan_Approved` | `1` = Approved, `0` = Rejected |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run the Project
```bash
git clone https://github.com/your-username/CreditWise-Loan-System.git
cd CreditWise-Loan-System
python main.py
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **ML Models:** (e.g., Logistic Regression, Random Forest, XGBoost)
- **Goal:** Binary Classification — Loan Approved (1) or Rejected (0)

---

## 📁 Project Structure

```
CreditWise-Loan-System/
│
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks for EDA & modelling
├── models/             # Saved ML models
├── src/                # Source code
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── README.md
└── requirements.txt
```

---

## 📈 Expected Outcomes

- A trained ML model that accurately predicts loan approval
- Reduced processing time for loan applications
- Fair and consistent decision-making across all applicant profiles

---

## 📄 License

This project is for educational and assessment purposes.
