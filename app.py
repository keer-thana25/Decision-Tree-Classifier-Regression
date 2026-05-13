# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io

# sklearn datasets
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing

# sklearn preprocessing
from sklearn.preprocessing import StandardScaler

# sklearn model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# sklearn tree models
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

# sklearn metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# suppress warnings
warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Decision Tree Project",
    page_icon="🌳",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1, h2, h3 {
    color: #0e1117;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    padding: 20px;
    color: gray;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌳 Navigation Menu")

menu = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📘 Theory",
        "🌸 Decision Tree Classifier",
        "🏠 Decision Tree Regression",
        "📐 Formula Page",
      
    ]
)

# =========================================================
# HOME PAGE
# =========================================================

if menu == "🏠 Home":

    st.title("🌳 Decision Tree Classifier and Decision Tree Regression")

    st.markdown("---")

    st.header("📌 Project Overview")

    st.write("""
    This mini project explains the working of **Decision Tree Algorithms**
    in Machine Learning.

    Decision Trees are one of the most important supervised learning algorithms.
    They are simple, powerful, easy to understand, and widely used in both
    classification and regression problems.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌸 Decision Tree Classifier")
        st.write("""
        A Decision Tree Classifier is used when the output is categorical.

        Example:
        - Spam or Not Spam
        - Disease or No Disease
        - Flower Type Prediction
        """)

    with col2:
        st.subheader("🏠 Decision Tree Regression")
        st.write("""
        A Decision Tree Regressor is used when the output is numerical.

        Example:
        - House Price Prediction
        - Temperature Prediction
        - Salary Prediction
        """)

    st.markdown("---")

    st.header("🌍 Real World Applications")

    applications = [
        "Medical Diagnosis",
        "Fraud Detection",
        "Loan Approval Systems",
        "Customer Segmentation",
        "House Price Prediction",
        "Weather Forecasting",
        "Recommendation Systems"
    ]

    for app in applications:
        st.write(f"✅ {app}")

    st.markdown("---")

    st.header("⭐ Why Decision Trees are Important")

    st.write("""
    - Easy to understand and interpret
    - Handles both numerical and categorical data
    - Requires less data preprocessing
    - Visual representation is simple
    - Useful for decision-making systems
    """)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("✅ Advantages")
        st.write("""
        - Simple and easy to understand
        - Works with nonlinear data
        - Requires less preprocessing
        - Good visualization
        - Fast prediction
        """)

    with col4:
        st.subheader("❌ Disadvantages")
        st.write("""
        - Can overfit easily
        - Sensitive to small data changes
        - Less accurate than ensemble models
        - Large trees become complex
        """)

    st.markdown("---")

    st.header("🔍 Classification vs Regression")

    comparison_df = pd.DataFrame({
        "Feature": [
            "Output Type",
            "Used For",
            "Example",
            "Algorithm"
        ],
        "Classification": [
            "Categorical",
            "Category Prediction",
            "Spam/Not Spam",
            "DecisionTreeClassifier"
        ],
        "Regression": [
            "Continuous Numerical",
            "Value Prediction",
            "House Price",
            "DecisionTreeRegressor"
        ]
    })

    st.table(comparison_df)

# =========================================================
# THEORY PAGE
# =========================================================

elif menu == "📘 Theory":

    st.title("📘 Detailed Theory of Decision Trees")

    st.markdown("---")

    st.header("A) What is Decision Tree?")

    st.write("""
    A Decision Tree is a supervised machine learning algorithm used for
    classification and regression tasks.

    It works like a flowchart structure where:
    - Root Node = Main decision point
    - Internal Nodes = Conditions/Questions
    - Leaf Nodes = Final output/result
    """)

    st.subheader("🌳 Structure of Decision Tree")

    st.write("""
    Root Node:
    - First node of the tree
    - Represents the entire dataset

    Internal Node:
    - Represents decision conditions
    - Splits the dataset

    Leaf Node:
    - Final output node
    - Represents prediction
    """)

    st.markdown("---")

    st.header("B) Decision Tree Classifier")

    st.write("""
    Decision Tree Classifier is used when the target variable is categorical.

    Example:
    - Predicting flower species
    - Disease classification
    - Email spam detection
    """)

    st.subheader("🌸 Why Iris Dataset?")

    st.write("""
    Iris dataset is suitable because:
    - It is simple and beginner-friendly
    - Contains 3 flower classes
    - Good for multiclass classification
    - Easy visualization
    """)

    st.markdown("---")

    st.header("C) Decision Tree Regression")

    st.write("""
    Decision Tree Regression predicts continuous numerical values.

    Example:
    - House price prediction
    - Sales prediction
    - Temperature prediction
    """)

    st.subheader("🏠 Why California Housing Dataset?")

    st.write("""
    California Housing dataset is suitable because:
    - Contains real-world numerical values
    - Ideal for regression tasks
    - Predicts median house prices
    """)

    st.markdown("---")

    st.header("D) Entropy")

    st.latex(r"Entropy(S) = -\sum p(x)\log_2 p(x)")

    st.write("""
    Entropy measures impurity or randomness in the dataset.

    Pure Node:
    - Contains only one class
    - Entropy = 0

    Impure Node:
    - Contains mixed classes
    - Entropy is high
    """)

    st.subheader("📌 High vs Low Entropy")

    st.write("""
    High Entropy:
    - Data is mixed
    - More uncertainty

    Low Entropy:
    - Data is pure
    - Less uncertainty
    """)

    st.markdown("---")

    st.header("E) Information Gain")

    st.latex(r"Information\ Gain = Entropy(parent) - Weighted\ Entropy(children)")

    st.write("""
    Information Gain measures how much information is gained after splitting.
    The feature with highest information gain is selected.
    """)

    st.markdown("---")

    st.header("F) Gini Index")

    st.latex(r"Gini = 1 - \sum p(x)^2")

    st.write("""
    Gini Index measures impurity.
    Lower Gini Index means better split.
    """)

    st.markdown("---")

    st.header("G) Pure vs Impure Nodes")

    st.write("""
    Pure Node:
    - All samples belong to one class
    - No confusion
    - Better prediction

    Impure Node:
    - Mixed classes
    - More confusion
    - Needs further splitting
    """)

    st.markdown("---")

    st.header("H) Overfitting vs Underfitting")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚠️ Overfitting")
        st.write("""
        - Model memorizes training data
        - Very complex tree
        - Poor performance on new data
        """)

    with col2:
        st.subheader("⚠️ Underfitting")
        st.write("""
        - Model is too simple
        - Cannot learn patterns properly
        - Low accuracy
        """)

    st.markdown("---")

    st.header("I) Pruning")

    st.write("""
    Pruning reduces tree size and prevents overfitting.
    """)

    st.subheader("Pre-Pruning")
    st.write("""
    Stops tree growth early using:
    - max_depth
    - min_samples_split
    """)

    st.subheader("Post-Pruning")
    st.write("""
    Removes unnecessary branches after tree construction.
    """)

    st.markdown("---")

    st.header("J) Hyperparameters")

    st.write("""
    max_depth:
    - Maximum depth of tree

    min_samples_split:
    - Minimum samples required to split

    min_samples_leaf:
    - Minimum samples in leaf node

    random_state:
    - Ensures same random output every time
    """)

    st.markdown("---")

    st.header("K) GridSearchCV")

    st.write("""
    GridSearchCV is used for hyperparameter tuning.

    It:
    - Tests multiple parameter combinations
    - Uses cross validation
    - Finds best parameters
    - Improves model performance
    """)

    st.markdown("---")

    st.header("L) Comparison Table")

    comparison = pd.DataFrame({
        "Feature": [
            "Output",
            "Dataset",
            "Evaluation",
            "Goal"
        ],
        "Classifier": [
            "Categorical",
            "Iris",
            "Accuracy",
            "Class Prediction"
        ],
        "Regression": [
            "Continuous Numerical",
            "California Housing",
            "MAE/MSE/RMSE",
            "Value Prediction"
        ]
    })

    st.table(comparison)

# =========================================================
# DECISION TREE CLASSIFIER
# =========================================================

elif menu == "🌸 Decision Tree Classifier":

    st.title("🌸 Decision Tree Classifier using Iris Dataset")

    st.markdown("---")

    # =====================================================
    # STEP 1
    # =====================================================

    st.header("Step 1: Load Iris Dataset")

    iris = load_iris()

    st.write("Dataset Loaded Successfully ✅")

    # =====================================================
    # STEP 2
    # =====================================================

    st.header("Step 2: Convert to DataFrame")

    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    st.dataframe(df.head())

    # =====================================================
    # STEP 3
    # =====================================================

    st.header("Step 3: First 5 Rows and Last 5 Rows")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("First 5 Rows")
        st.dataframe(df.head())

    with col2:
        st.subheader("Last 5 Rows")
        st.dataframe(df.tail())

    # =====================================================
    # STEP 4
    # =====================================================

    st.header("Step 4: Dataset Information")

    st.write("Shape of Dataset:", df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Dataset Info")

    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()

    st.text(s)

    # =====================================================
    # STEP 5
    # =====================================================

    st.header("Step 5: Outlier Detection using Boxplot")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df.drop("target", axis=1), ax=ax)
    st.pyplot(fig)

    # =====================================================
    # STEP 6
    # =====================================================

    st.header("Step 6: Handle Outliers using IQR Method")

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)

    IQR = Q3 - Q1

    df_clean = df[~((df < (Q1 - 1.5 * IQR)) |
                    (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    st.write("Original Shape:", df.shape)
    st.write("After Removing Outliers:", df_clean.shape)

    # =====================================================
    # STEP 7
    # =====================================================

    st.header("Step 7: Visualization After Removing Outliers")

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_clean.drop("target", axis=1), ax=ax2)
    st.pyplot(fig2)

    # =====================================================
    # STEP 8
    # =====================================================

    st.header("Step 8: Iris Target")

    st.write(iris.target_names)

    # =====================================================
    # STEP 9
    # =====================================================

    st.header("Step 9: Feature Scaling using StandardScaler")

    X = df_clean.drop("target", axis=1)
    y = df_clean["target"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    st.write("Feature Scaling Completed ✅")

    # =====================================================
    # STEP 10
    # =====================================================

    st.header("Step 10: Train Test Split")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    st.write("Training Data Shape:", X_train.shape)
    st.write("Testing Data Shape:", X_test.shape)

    # =====================================================
    # STEP 11
    # =====================================================

    st.header("Step 11: Train DecisionTreeClassifier")

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    st.write("Model Trained Successfully ✅")

    # =====================================================
    # STEP 12
    # =====================================================

    st.header("Step 12: Model Parameters")

    st.write(model.get_params())

    # =====================================================
    # STEP 13
    # =====================================================

    st.header("Step 13: Warnings Suppressed")

    st.write("Warnings imported and suppressed successfully ✅")

    # =====================================================
    # STEP 14
    # =====================================================

    st.header("Step 14: Apply GridSearchCV")

    param_grid = {
        "max_depth": [2, 3, 4, 5],
        "min_samples_split": [2, 3, 4],
        "min_samples_leaf": [1, 2, 3]
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5
    )

    grid.fit(X_train, y_train)

    # =====================================================
    # STEP 15
    # =====================================================

    st.header("Step 15: Best Parameters and Best Score")

    st.write("Best Parameters:", grid.best_params_)
    st.write("Best Score:", grid.best_score_)

    # =====================================================
    # STEP 16
    # =====================================================

    st.header("Step 16: Prediction")

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    st.write("Predictions Completed ✅")

    # =====================================================
    # STEP 17
    # =====================================================

    st.header("Step 17: Evaluation")

    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Confusion Matrix")
    st.write(cm)

    st.subheader("Classification Report")

    report = classification_report(y_test, y_pred)
    st.text(report)

    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Accuracy Score")
    st.write(accuracy)

    # =====================================================
    # STEP 18
    # =====================================================

    st.header("Step 18: Visualize Confusion Matrix")

    fig3, ax3 = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax3
    )

    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    st.pyplot(fig3)

    # =====================================================
    # STEP 19 — CHANGED: Multi-depth Decision Tree Visualization
    # =====================================================

    st.header("Step 19: Visualize Decision Tree")

    st.info("🔍 Use the slider below to explore the tree at different depths. Lower depth = simpler and clearer view.")

    viz_depth = st.slider(
        "Select Max Depth for Visualization",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        key="clf_viz_depth"
    )

    st.subheader(f"Decision Tree Visualization — Max Depth: {viz_depth}")

    fig4, ax4 = plt.subplots(figsize=(20, 10))

    plot_tree(
        best_model,
        max_depth=viz_depth,
        filled=True,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        rounded=True,
        fontsize=10,
        ax=ax4
    )

    ax4.set_title(f"Decision Tree Classifier — Showing Depth {viz_depth}", fontsize=14)

    st.pyplot(fig4)

    st.caption(
        "💡 Tip: Depth 1–2 shows the most important splits. "
        "Increase depth gradually to see more detail. "
        f"The trained model's actual best depth is: {best_model.get_depth()}"
    )

    # =====================================================
    # STEP 20
    # =====================================================

    st.header("Step 20: Result Explanation")

    st.success("""
    The Decision Tree Classifier successfully classified the Iris flowers.

    - GridSearchCV helped find the best parameters.
    - Confusion Matrix shows prediction performance.
    - Accuracy Score indicates how well the model performed.
    - Decision Tree visualization shows decision rules clearly.
    """)

# =========================================================
# DECISION TREE REGRESSION
# =========================================================

elif menu == "🏠 Decision Tree Regression":

    st.title("🏠 Decision Tree Regression using California Housing Dataset")

    st.markdown("---")

    # =====================================================
    # STEP 1
    # =====================================================

    st.header("Step 1: Load California Housing Dataset")

    housing = fetch_california_housing()

    st.write("Dataset Loaded Successfully ✅")

    # =====================================================
    # STEP 2
    # =====================================================

    st.header("Step 2: Convert to DataFrame")

    df = pd.DataFrame(housing.data, columns=housing.feature_names)

    df["target"] = housing.target

    st.dataframe(df.head())

    # =====================================================
    # STEP 3
    # =====================================================

    st.header("Step 3: First 5 Rows and Last 5 Rows")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("First 5 Rows")
        st.dataframe(df.head())

    with col2:
        st.subheader("Last 5 Rows")
        st.dataframe(df.tail())

    # =====================================================
    # STEP 4
    # =====================================================

    st.header("Step 4: Dataset Information")

    st.write("Shape:", df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()

    st.subheader("Info()")
    st.text(s)

    # =====================================================
    # STEP 5
    # =====================================================

    st.header("Step 5: Outlier Detection using Boxplot")

    fig5, ax5 = plt.subplots(figsize=(15, 7))

    sns.boxplot(data=df.drop("target", axis=1), ax=ax5)

    st.pyplot(fig5)

    # =====================================================
    # STEP 6
    # =====================================================

    st.header("Step 6: Handle Outliers using IQR Method")

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)

    IQR = Q3 - Q1

    df_clean = df[~((df < (Q1 - 1.5 * IQR)) |
                    (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    st.write("Original Shape:", df.shape)
    st.write("After Removing Outliers:", df_clean.shape)

    # =====================================================
    # STEP 7
    # =====================================================

    st.header("Step 7: Visualization After Removing Outliers")

    fig6, ax6 = plt.subplots(figsize=(15, 7))

    sns.boxplot(data=df_clean.drop("target", axis=1), ax=ax6)

    st.pyplot(fig6)

    # =====================================================
    # STEP 8
    # =====================================================

    st.header("Step 8: Feature Scaling using StandardScaler")

    X = df_clean.drop("target", axis=1)
    y = df_clean["target"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    st.write("Scaling Completed ✅")

    # =====================================================
    # STEP 9
    # =====================================================

    st.header("Step 9: Train Test Split")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    st.write("Train Shape:", X_train.shape)
    st.write("Test Shape:", X_test.shape)

    # =====================================================
    # STEP 10
    # =====================================================

    st.header("Step 10: Train DecisionTreeRegressor")

    regressor = DecisionTreeRegressor(random_state=42)

    regressor.fit(X_train, y_train)

    st.write("Model Trained Successfully ✅")

    # =====================================================
    # STEP 11
    # =====================================================

    st.header("Step 11: Model Parameters")

    st.write(regressor.get_params())

    # =====================================================
    # STEP 12
    # =====================================================

    st.header("Step 12: Apply GridSearchCV")

    param_grid = {
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid,
        cv=5
    )

    grid.fit(X_train, y_train)

    # =====================================================
    # STEP 13
    # =====================================================

    st.header("Step 13: Best Parameters and Best Score")

    st.write("Best Parameters:", grid.best_params_)
    st.write("Best Score:", grid.best_score_)

    # =====================================================
    # STEP 14
    # =====================================================

    st.header("Step 14: Prediction")

    best_regressor = grid.best_estimator_

    y_pred = best_regressor.predict(X_test)

    st.write("Prediction Completed ✅")

    # =====================================================
    # STEP 15
    # =====================================================

    st.header("Step 15: Evaluation Metrics")

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    st.write("Mean Absolute Error (MAE):", mae)
    st.write("Mean Squared Error (MSE):", mse)
    st.write("Root Mean Squared Error (RMSE):", rmse)
    st.write("R² Score:", r2)

    # =====================================================
    # STEP 16
    # =====================================================

    st.header("Step 16: Actual vs Predicted Graph")

    fig7, ax7 = plt.subplots(figsize=(10, 6))

    ax7.scatter(y_test[:200], y_pred[:200])

    ax7.set_xlabel("Actual Values")
    ax7.set_ylabel("Predicted Values")

    ax7.set_title("Actual vs Predicted")

    st.pyplot(fig7)

    # =====================================================
    # STEP 17 — CHANGED: Multi-depth Decision Tree Visualization
    # =====================================================

    st.header("Step 17: Visualize Decision Tree Regression")

    st.info("🔍 Use the slider below to explore the tree at different depths. Lower depth = fewer nodes and clearer view.")

    viz_depth = st.slider(
        "Select Max Depth for Visualization",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        key="reg_viz_depth"
    )

    st.subheader(f"Decision Tree Visualization — Max Depth: {viz_depth}")

    fig8, ax8 = plt.subplots(figsize=(20, 10))

    plot_tree(
        best_regressor,
        max_depth=viz_depth,
        filled=True,
        feature_names=housing.feature_names,
        fontsize=8,
        ax=ax8
    )

    ax8.set_title(f"Decision Tree Regressor — Showing Depth {viz_depth}", fontsize=14)

    st.pyplot(fig8)

    st.caption(
        "💡 Tip: Start at Depth 1–2 to see the most impactful splits clearly. "
        "Increase depth to explore more nodes. "
        f"The trained model's actual best depth is: {best_regressor.get_depth()}"
    )

    # =====================================================
    # STEP 18
    # =====================================================

    st.header("Step 18: Result Explanation")

    st.success("""
    The Decision Tree Regressor successfully predicted housing prices.

    - MAE measures average error.
    - MSE measures squared error.
    - RMSE gives root error value.
    - R² Score measures model performance.
    - GridSearchCV optimized the model.
    """)

# =========================================================
# FORMULA PAGE
# =========================================================

elif menu == "📐 Formula Page":

    st.title("📐 Important Formulas")

    st.markdown("---")

    st.header("Entropy")

    st.latex(r"Entropy(S) = -\sum p(x)\log_2 p(x)")

    st.markdown("---")

    st.header("Information Gain")

    st.latex(r"IG = Entropy(parent) - Weighted\ Entropy(children)")

    st.markdown("---")

    st.header("Gini Index")

    st.latex(r"Gini = 1 - \sum p(x)^2")

    st.markdown("---")

    st.header("Accuracy")

    st.latex(r"Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}")

    st.markdown("---")

    st.header("MAE")

    st.latex(r"MAE = \frac{1}{n}\sum |y - \hat{y}|")

    st.markdown("---")

    st.header("MSE")

    st.latex(r"MSE = \frac{1}{n}\sum (y - \hat{y})^2")

    st.markdown("---")

    st.header("RMSE")

    st.latex(r"RMSE = \sqrt{MSE}")

    st.markdown("---")

    st.header("R² Score")

    st.latex(r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}}")



# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    <h4>🌳 Decision Tree Classifier and Decision Tree Regression</h4>
    <p>Machine Learning Mini Project using Streamlit</p>
    <p>Developed for College Mini Project Presentation</p>
</div>
""", unsafe_allow_html=True)
