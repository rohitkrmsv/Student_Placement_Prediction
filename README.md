# 🎓 Student Placement Prediction System

A Machine Learning web application that predicts whether a student is likely to get placed based on academic performance, technical skills, communication skills, internships, and projects.

## 🚀 Features

- Student placement prediction
- Placement probability
- Multiple machine learning model comparison
- Logistic Regression
- Random Forest
- Gradient Boosting
- Model performance evaluation
- Feature impact explanation
- Personalized improvement suggestions
- Interactive Streamlit web interface

## 🧠 Machine Learning Models

The project compares three classification algorithms:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

### Model Performance

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 92.5% |
| Random Forest | 91.5% |
| Gradient Boosting | 91.5% |

## 📊 Input Features

The model uses the following student attributes:

- CGPA
- Attendance
- Coding Score
- Aptitude Score
- Communication Score
- Number of Internships
- Number of Projects

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## 📁 Project Structure

```text
Student_Placement_Prediction/
│
├── data/
│   ├── students.csv
│   └── students_large.csv
│
├── app.py
├── train_model.py
├── test.py
├── placement_model.pkl
├── requirements.txt
├── README.md
└── .gitignore