# 🌳 Decision Tree Classifier and Decision Tree Regression

A professional **Machine Learning Streamlit Web Application** that demonstrates the implementation of **Decision Tree Classification** using the **Iris Dataset** and **Decision Tree Regression** using the **California Housing Dataset**.

This project provides both **practical implementation** and **theoretical understanding** of Decision Trees, including concepts such as **Entropy, Gini Index, Information Gain, Pure vs Impure Nodes, Pruning, GridSearchCV, Hyperparameter Tuning, and Model Evaluation**.

---

## 🚀 Live Demo

🔗 **Deployed Application:**  
🌐 Live App: https://decision-tree-classifier-regression.streamlit.app/

---

## 📌 Project Overview

This project is designed to provide a **complete understanding of Decision Trees in Machine Learning** through:

- 🌸 **Decision Tree Classification**
- 🏠 **Decision Tree Regression**
- 📊 Data preprocessing
- 📈 Outlier detection & removal
- ⚙️ Feature scaling
- 🔍 Hyperparameter tuning using GridSearchCV
- 📉 Data visualization
- 📋 Performance evaluation
- 🌳 Decision Tree visualization
- 📚 Theory and formulas

The application includes a **professional UI built using Streamlit** to improve usability, understanding, and presentation.

---

## ✨ Features

### 🌸 Decision Tree Classifier (Iris Dataset)

Implemented using the **Iris Dataset** to classify flower species.

### Workflow

✔ Load Dataset  
✔ Display First 5 & Last 5 Rows  
✔ Dataset Information (`shape`, `info()`)  
✔ Missing Value Detection  
✔ Outlier Detection using Box Plot  
✔ Outlier Handling using IQR Method  
✔ Feature Scaling using StandardScaler  
✔ Train-Test Split  
✔ DecisionTreeClassifier Training  
✔ Hyperparameter Tuning using GridSearchCV  
✔ Best Parameters & Best Score  
✔ Prediction  
✔ Confusion Matrix  
✔ Classification Report  
✔ Accuracy Score  
✔ Decision Tree Visualization

---

### 🏠 Decision Tree Regression (California Housing Dataset)

Implemented using the **California Housing Dataset** to predict house prices.

### Workflow

✔ Load Dataset  
✔ Display First 5 & Last 5 Rows  
✔ Dataset Information (`shape`, `info()`)  
✔ Missing Value Detection  
✔ Outlier Detection using Box Plot  
✔ Outlier Handling using IQR Method  
✔ Feature Scaling using StandardScaler  
✔ Train-Test Split  
✔ DecisionTreeRegressor Training  
✔ Hyperparameter Tuning using GridSearchCV  
✔ Best Parameters & Best Score  
✔ Prediction  
✔ Mean Absolute Error (MAE)  
✔ Mean Squared Error (MSE)  
✔ Root Mean Squared Error (RMSE)  
✔ R² Score  
✔ Actual vs Predicted Visualization  
✔ Decision Tree Visualization

---

## 🧠 Decision Tree Concepts Covered

### 1. What is a Decision Tree?

A **Decision Tree** is a supervised machine learning algorithm used for both:

- **Classification**
- **Regression**

It works like a tree structure where:

- **Root Node** → Starting point of decision making  
- **Internal Nodes** → Decision rules/features  
- **Leaf Nodes** → Final prediction/output

---

### 2. Entropy

Entropy measures the **impurity or randomness** in a dataset.

### Formula

```math
Entropy(S) = - \sum p(x) \log_2 p(x)
```

### Explanation

- **High Entropy** → Impure data
- **Low Entropy** → Pure data

---

### 3. Information Gain

Information Gain helps determine the **best feature for splitting** data.

### Formula

```math
IG(S,A)=Entropy(S)-Weighted\ Average\ Entropy
```

---

### 4. Gini Index

Gini Index measures **impurity in classification problems**.

### Formula

```math
Gini = 1 - \sum p_i^2
```

---

### 5. Pure vs Impure Nodes

### Pure Node
Contains data from **only one class**.

### Impure Node
Contains data from **multiple classes**.

---

### 6. Pruning

Pruning is used to **reduce overfitting**.

### Types of Pruning

- **Pre-Pruning**
- **Post-Pruning**

---

### 7. GridSearchCV

GridSearchCV helps find the **best hyperparameters automatically**.

### Hyperparameters Used

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

---

## 📊 Evaluation Metrics

### Classification Metrics

- Confusion Matrix
- Classification Report
- Accuracy Score

### Regression Metrics

- Mean Absolute Error (**MAE**)
- Mean Squared Error (**MSE**)
- Root Mean Squared Error (**RMSE**)
- R² Score

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Framework

- Streamlit

### Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📂 Project Structure

```text
project-folder/
│── app.py
│── README.md
│── requirements.txt
│── decision_tree_classifier.ipynb
│── decision_tree_regression.ipynb
```

---

## 📁 Files Description

### `app.py`

Main Streamlit application containing:

- Decision Tree Theory
- Classifier Implementation
- Regression Implementation
- Visualizations
- GridSearchCV
- Evaluation Metrics

### `decision_tree_classifier.ipynb`

Notebook for **Decision Tree Classification using Iris Dataset**.

Includes:

- Data preprocessing
- Outlier detection
- IQR Method
- StandardScaler
- GridSearchCV
- Confusion Matrix
- Classification Report
- Accuracy Score
- Decision Tree Visualization

### `decision_tree_regression.ipynb`

Notebook for **Decision Tree Regression using California Housing Dataset**.

Includes:

- Data preprocessing
- Outlier detection
- IQR Method
- Feature scaling
- GridSearchCV
- MAE, MSE, RMSE, R² Score
- Actual vs Predicted Visualization
- Decision Tree Visualization

### `requirements.txt`

Contains all required dependencies to run the project.

### `README.md`

Project documentation including:

- Project Overview
- Installation Guide
- Features
- Workflow
- Deployment Link
- Technical Details

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### Navigate to Project Folder

```bash
cd project-folder
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Run the Streamlit application using:

```bash
streamlit run app.py
```

---

## 📷 Screenshots

Add screenshots of your Streamlit application here.

### 🏠 Home Page

(Add Screenshot)

### 🌸 Decision Tree Classifier

(Add Screenshot)

### 🏠 Decision Tree Regression

(Add Screenshot)

---

## 🔮 Future Enhancements

- Add more Machine Learning algorithms
- Improve UI/UX design
- Add downloadable reports
- Add model comparison dashboard
- Include real-time prediction system

---

## 🎓 Learning Outcomes

Through this project, we learned:

- Difference between Classification & Regression
- Working of Decision Trees
- Entropy & Information Gain
- Hyperparameter tuning using GridSearchCV
- Feature scaling
- Outlier handling
- Model evaluation techniques
- Decision Tree visualization

---

## 👨‍💻 Author

**Keerthana**



---


