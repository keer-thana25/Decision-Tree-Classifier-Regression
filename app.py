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
# CONSTANTS
# =========================================================

USD_TO_INR = 83.5  # 1 USD = 83.5 INR (approximate)

def format_inr(amount):
    """Format amount in Indian Rupees with crore/lakh notation."""
    if amount >= 1_00_00_000:  # 1 crore
        crores = amount / 1_00_00_000
        return f"Rs.{crores:.2f} Cr"
    elif amount >= 1_00_000:   # 1 lakh
        lakhs = amount / 1_00_000
        return f"Rs.{lakhs:.2f} L"
    else:
        return f"Rs.{amount:,.0f}"

def format_inr_symbol(amount):
    """Format with rupee symbol for HTML display."""
    if amount >= 1_00_00_000:
        crores = amount / 1_00_00_000
        return f"&#8377;{crores:.2f} Cr"
    elif amount >= 1_00_000:
        lakhs = amount / 1_00_000
        return f"&#8377;{lakhs:.2f} L"
    else:
        return f"&#8377;{amount:,.0f}"

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

.predict-box {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    border: 2px solid #4CAF50;
    border-radius: 15px;
    padding: 25px;
    margin-top: 20px;
}

/* ── Big result card ── */
.result-box {
    background: linear-gradient(135deg, #0D1B4B, #1a237e, #283593);
    border-radius: 24px;
    padding: 40px 30px 32px 30px;
    margin-top: 20px;
    text-align: center;
    box-shadow: 0 10px 40px rgba(26,35,126,0.35);
}

.result-label {
    color: #90CAF9;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.result-price-inr {
    color: #FFD54F;
    font-size: 64px;
    font-weight: 900;
    letter-spacing: 1px;
    text-shadow: 0 3px 12px rgba(0,0,0,0.5);
    margin: 8px 0 4px 0;
    line-height: 1.1;
}

.result-price-full {
    color: #E3F2FD;
    font-size: 20px;
    font-weight: 500;
    margin-top: 6px;
    letter-spacing: 0.5px;
}

.result-price-sub {
    color: #90A4AE;
    font-size: 14px;
    margin-top: 10px;
}

/* ── Metric mini-cards ── */
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
    box-shadow: 0 2px 14px rgba(0,0,0,0.09);
    border-left: 5px solid #1565C0;
    height: 100%;
}

.metric-label {
    color: #607D8B;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.metric-value {
    color: #1565C0;
    font-size: 28px;
    font-weight: 900;
    margin-top: 6px;
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
    This mini project explains the working of **Decision Tree Algorithms** in Machine Learning.
    Decision Trees are one of the most important supervised learning algorithms.
    They are simple, powerful, easy to understand, and widely used in both
    classification and regression problems.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌸 Decision Tree Classifier")
        st.write("""
        A Decision Tree Classifier is used when the output is categorical.\n
        Example:\n- Spam or Not Spam\n- Disease or No Disease\n- Flower Type Prediction
        """)
    with col2:
        st.subheader("🏠 Decision Tree Regression")
        st.write("""
        A Decision Tree Regressor is used when the output is numerical.\n
        Example:\n- House Price Prediction\n- Temperature Prediction\n- Salary Prediction
        """)

    st.markdown("---")
    st.header("🌍 Real World Applications")
    for app in ["Medical Diagnosis","Fraud Detection","Loan Approval Systems",
                 "Customer Segmentation","House Price Prediction","Weather Forecasting","Recommendation Systems"]:
        st.write(f"✅ {app}")

    st.markdown("---")
    st.header("⭐ Why Decision Trees are Important")
    st.write("""
    - Easy to understand and interpret\n- Handles both numerical and categorical data
    - Requires less data preprocessing\n- Visual representation is simple
    - Useful for decision-making systems
    """)
    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("✅ Advantages")
        st.write("- Simple and easy to understand\n- Works with nonlinear data\n- Requires less preprocessing\n- Good visualization\n- Fast prediction")
    with col4:
        st.subheader("❌ Disadvantages")
        st.write("- Can overfit easily\n- Sensitive to small data changes\n- Less accurate than ensemble models\n- Large trees become complex")

    st.markdown("---")
    st.header("🔍 Classification vs Regression")
    st.table(pd.DataFrame({
        "Feature": ["Output Type","Used For","Example","Algorithm"],
        "Classification": ["Categorical","Category Prediction","Spam/Not Spam","DecisionTreeClassifier"],
        "Regression": ["Continuous Numerical","Value Prediction","House Price","DecisionTreeRegressor"]
    }))

# =========================================================
# THEORY PAGE
# =========================================================

elif menu == "📘 Theory":

    st.title("📘 Detailed Theory of Decision Trees")
    st.markdown("---")

    st.header("A) What is Decision Tree?")
    st.write("""
    A Decision Tree is a supervised machine learning algorithm used for classification and regression tasks.
    It works like a flowchart structure where:\n
    - Root Node = Main decision point\n- Internal Nodes = Conditions/Questions\n- Leaf Nodes = Final output/result
    """)

    st.header("B) Decision Tree Classifier")
    st.write("Decision Tree Classifier is used when the target variable is categorical.")
    st.subheader("🌸 Why Iris Dataset?")
    st.write("Simple, beginner-friendly, 3 flower classes, good for multiclass classification.")

    st.markdown("---")
    st.header("C) Decision Tree Regression")
    st.write("Decision Tree Regression predicts continuous numerical values.")
    st.subheader("🏠 Why California Housing Dataset?")
    st.write("Contains real-world numerical values, ideal for regression, predicts median house prices.")

    st.markdown("---")
    st.header("D) Entropy")
    st.latex(r"Entropy(S) = -\sum p(x)\log_2 p(x)")
    st.write("Entropy measures impurity or randomness. Pure Node = 0, Impure Node = high entropy.")

    st.markdown("---")
    st.header("E) Information Gain")
    st.latex(r"Information\ Gain = Entropy(parent) - Weighted\ Entropy(children)")

    st.markdown("---")
    st.header("F) Gini Index")
    st.latex(r"Gini = 1 - \sum p(x)^2")
    st.write("Lower Gini Index means better split.")

    st.markdown("---")
    st.header("G) Overfitting vs Underfitting")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚠️ Overfitting")
        st.write("Model memorizes training data. Very complex tree. Poor on new data.")
    with col2:
        st.subheader("⚠️ Underfitting")
        st.write("Model is too simple. Cannot learn patterns. Low accuracy.")

    st.markdown("---")
    st.header("H) Pruning")
    st.write("Pre-Pruning: Stops early using max_depth, min_samples_split.")
    st.write("Post-Pruning: Removes unnecessary branches after construction.")

    st.markdown("---")
    st.header("I) GridSearchCV")
    st.write("Tests multiple parameter combinations using cross-validation to find best params.")

    st.markdown("---")
    st.header("J) Comparison Table")
    st.table(pd.DataFrame({
        "Feature": ["Output","Dataset","Evaluation","Goal"],
        "Classifier": ["Categorical","Iris","Accuracy","Class Prediction"],
        "Regression": ["Continuous Numerical","California Housing","MAE/MSE/RMSE","Value Prediction"]
    }))

# =========================================================
# DECISION TREE CLASSIFIER
# =========================================================

elif menu == "🌸 Decision Tree Classifier":

    st.title("🌸 Decision Tree Classifier using Iris Dataset")
    st.markdown("---")

    st.header("Step 1: Load Iris Dataset")
    iris = load_iris()
    st.write("Dataset Loaded Successfully ✅")

    st.header("Step 2: Convert to DataFrame")
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    st.dataframe(df.head())

    st.header("Step 3: First 5 and Last 5 Rows")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("First 5 Rows")
        st.dataframe(df.head())
    with col2:
        st.subheader("Last 5 Rows")
        st.dataframe(df.tail())

    st.header("Step 4: Dataset Information")
    st.write("Shape:", df.shape)
    st.write(df.isnull().sum())
    buf = io.StringIO(); df.info(buf=buf); st.text(buf.getvalue())

    st.header("Step 5: Outlier Detection")
    fig, ax = plt.subplots(figsize=(12,6)); sns.boxplot(data=df.drop("target",axis=1),ax=ax); st.pyplot(fig)

    st.header("Step 6: Remove Outliers (IQR)")
    Q1,Q3 = df.quantile(0.25), df.quantile(0.75); IQR = Q3-Q1
    df_clean = df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
    st.write("Original:",df.shape,"→ Cleaned:",df_clean.shape)

    st.header("Step 7: After Outlier Removal")
    fig2,ax2=plt.subplots(figsize=(12,6)); sns.boxplot(data=df_clean.drop("target",axis=1),ax=ax2); st.pyplot(fig2)

    st.header("Step 8: Iris Target Names")
    st.write(iris.target_names)

    st.header("Step 9: Feature Scaling")
    X = df_clean.drop("target",axis=1); y = df_clean["target"]
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    st.write("Feature Scaling Completed ✅")

    st.header("Step 10: Train Test Split")
    X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)
    st.write("Train:",X_train.shape,"| Test:",X_test.shape)

    st.header("Step 11: Train Model")
    model = DecisionTreeClassifier(random_state=42); model.fit(X_train,y_train)
    st.write("Model Trained ✅")

    st.header("Step 12: Model Parameters")
    st.write(model.get_params())

    st.header("Step 13: Warnings Suppressed")
    st.write("Warnings suppressed ✅")

    st.header("Step 14: GridSearchCV")
    param_grid={"max_depth":[2,3,4,5],"min_samples_split":[2,3,4],"min_samples_leaf":[1,2,3]}
    grid=GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid,cv=5)
    grid.fit(X_train,y_train)

    st.header("Step 15: Best Parameters")
    st.write("Best Params:",grid.best_params_); st.write("Best Score:",grid.best_score_)

    st.header("Step 16: Prediction")
    best_model=grid.best_estimator_; y_pred=best_model.predict(X_test)
    st.write("Predictions Completed ✅")

    st.header("Step 17: Evaluation")
    cm=confusion_matrix(y_test,y_pred)
    st.write(cm); st.text(classification_report(y_test,y_pred))
    st.write("Accuracy:",accuracy_score(y_test,y_pred))

    st.header("Step 18: Confusion Matrix Heatmap")
    fig3,ax3=plt.subplots(figsize=(6,5))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax3)
    ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual"); st.pyplot(fig3)

    st.header("Step 19: Decision Tree Visualization")
    viz_depth=st.slider("Max Depth",1,10,3,key="clf_viz_depth")
    fig4,ax4=plt.subplots(figsize=(20,10))
    plot_tree(best_model,max_depth=viz_depth,filled=True,feature_names=iris.feature_names,
              class_names=iris.target_names,rounded=True,fontsize=10,ax=ax4)
    ax4.set_title(f"Decision Tree — Depth {viz_depth}"); st.pyplot(fig4)
    st.caption(f"Actual model depth: {best_model.get_depth()}")

    st.header("Step 20: Result")
    st.success("Decision Tree Classifier successfully classified Iris flowers using GridSearchCV.")

    # ──────────────────────────────────────────────────────
    # STEP 21: Predict Your Own Iris Flower
    # ──────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Step 21: 🔮 Predict Your Own Iris Flower")
    st.markdown('<div class="predict-box"><h4 style="color:#2E7D32;">🌸 Enter flower measurements to predict species!</h4></div>', unsafe_allow_html=True)
    st.write("")

    idf = pd.DataFrame(iris.data, columns=iris.feature_names)
    col_a, col_b = st.columns(2)
    with col_a:
        sl = st.number_input("Sepal Length (cm)", float(idf.min()["sepal length (cm)"]), float(idf.max()["sepal length (cm)"]), float(round(idf.mean()["sepal length (cm)"],1)), 0.1, key="clf_sl")
        sw = st.number_input("Sepal Width (cm)",  float(idf.min()["sepal width (cm)"]),  float(idf.max()["sepal width (cm)"]),  float(round(idf.mean()["sepal width (cm)"],1)),  0.1, key="clf_sw")
    with col_b:
        pl = st.number_input("Petal Length (cm)", float(idf.min()["petal length (cm)"]), float(idf.max()["petal length (cm)"]), float(round(idf.mean()["petal length (cm)"],1)), 0.1, key="clf_pl")
        pw = st.number_input("Petal Width (cm)",  float(idf.min()["petal width (cm)"]),  float(idf.max()["petal width (cm)"]),  float(round(idf.mean()["petal width (cm)"],1)),  0.1, key="clf_pw")

    if st.button("🌸 Predict Flower Species", key="clf_predict_btn"):
        inp = scaler.transform([[sl,sw,pl,pw]])
        pred = best_model.predict(inp)[0]
        proba = best_model.predict_proba(inp)[0]
        name = iris.target_names[pred]
        emoji = {"setosa":"🌺","versicolor":"🌸","virginica":"🌼"}.get(name,"🌸")
        st.markdown(f'<div class="result-box"><div class="result-label">Predicted Flower Species</div><div class="result-price-inr">{emoji} Iris {name.upper()} {emoji}</div><div class="result-price-full">Confidence: {max(proba)*100:.1f}%</div></div>', unsafe_allow_html=True)
        st.write("")
        prob_df = pd.DataFrame({"Species":iris.target_names,"Probability (%)": [round(p*100,2) for p in proba]})
        st.dataframe(prob_df, use_container_width=True)
        fig_p,ax_p=plt.subplots(figsize=(7,4))
        ax_p.bar(iris.target_names, proba*100, color=["#FF7043","#42A5F5","#66BB6A"], edgecolor="black")
        for i,v in enumerate(proba*100): ax_p.text(i,v+1.5,f"{v:.1f}%",ha="center",fontweight="bold")
        ax_p.set_ylim(0,110); ax_p.set_title("Prediction Probability per Class"); st.pyplot(fig_p)
        st.success(f"✅ Iris **{name}** — {max(proba)*100:.1f}% confidence")

# =========================================================
# DECISION TREE REGRESSION
# =========================================================

elif menu == "🏠 Decision Tree Regression":

    st.title("🏠 Decision Tree Regression using California Housing Dataset")
    st.markdown("---")

    st.header("Step 1: Load Dataset")
    housing = fetch_california_housing()
    st.write("Dataset Loaded ✅")

    st.header("Step 2: DataFrame")
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["target"] = housing.target
    st.dataframe(df.head())

    st.header("Step 3: First & Last 5 Rows")
    c1,c2=st.columns(2)
    with c1: st.subheader("First 5"); st.dataframe(df.head())
    with c2: st.subheader("Last 5");  st.dataframe(df.tail())

    st.header("Step 4: Info")
    st.write("Shape:",df.shape); st.write(df.isnull().sum())
    buf=io.StringIO(); df.info(buf=buf); st.text(buf.getvalue())

    st.header("Step 5: Outlier Boxplot")
    fig5,ax5=plt.subplots(figsize=(15,7)); sns.boxplot(data=df.drop("target",axis=1),ax=ax5); st.pyplot(fig5)

    st.header("Step 6: Remove Outliers")
    Q1,Q3=df.quantile(0.25),df.quantile(0.75); IQR=Q3-Q1
    df_clean=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
    st.write("Original:",df.shape,"→ Cleaned:",df_clean.shape)

    st.header("Step 7: After Removal")
    fig6,ax6=plt.subplots(figsize=(15,7)); sns.boxplot(data=df_clean.drop("target",axis=1),ax=ax6); st.pyplot(fig6)

    st.header("Step 8: Feature Scaling")
    X=df_clean.drop("target",axis=1); y=df_clean["target"]
    scaler=StandardScaler(); X_scaled=scaler.fit_transform(X)
    st.write("Scaling Completed ✅")

    st.header("Step 9: Train Test Split")
    X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)
    st.write("Train:",X_train.shape,"| Test:",X_test.shape)

    st.header("Step 10: Train Regressor")
    regressor=DecisionTreeRegressor(random_state=42); regressor.fit(X_train,y_train)
    st.write("Model Trained ✅")

    st.header("Step 11: Parameters")
    st.write(regressor.get_params())

    st.header("Step 12: GridSearchCV")
    param_grid={"max_depth":[3,5,7],"min_samples_split":[2,5,10],"min_samples_leaf":[1,2,4]}
    grid=GridSearchCV(DecisionTreeRegressor(random_state=42),param_grid,cv=5)
    grid.fit(X_train,y_train)

    st.header("Step 13: Best Parameters")
    st.write("Best Params:",grid.best_params_); st.write("Best Score:",grid.best_score_)

    st.header("Step 14: Prediction")
    best_regressor=grid.best_estimator_; y_pred=best_regressor.predict(X_test)
    st.write("Prediction Completed ✅")

    st.header("Step 15: Evaluation Metrics")
    mae=mean_absolute_error(y_test,y_pred); mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse); r2=r2_score(y_test,y_pred)
    st.write("MAE:",mae); st.write("MSE:",mse); st.write("RMSE:",rmse); st.write("R²:",r2)

    st.header("Step 16: Actual vs Predicted")
    fig7,ax7=plt.subplots(figsize=(10,6))
    ax7.scatter(y_test[:200],y_pred[:200]); ax7.set_xlabel("Actual"); ax7.set_ylabel("Predicted"); ax7.set_title("Actual vs Predicted"); st.pyplot(fig7)

    st.header("Step 17: Tree Visualization")
    viz_depth=st.slider("Max Depth",1,10,3,key="reg_viz_depth")
    fig8,ax8=plt.subplots(figsize=(20,10))
    plot_tree(best_regressor,max_depth=viz_depth,filled=True,feature_names=housing.feature_names,fontsize=8,ax=ax8)
    ax8.set_title(f"Decision Tree Regressor — Depth {viz_depth}"); st.pyplot(fig8)
    st.caption(f"Actual model depth: {best_regressor.get_depth()}")

    st.header("Step 18: Result")
    st.success("Decision Tree Regressor successfully predicted housing prices.")

    # ──────────────────────────────────────────────────────
    # STEP 19: Predict Your Own House Price — IN INR ₹
    # ──────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Step 19: 🔮 Predict Your Own House Price")

    st.markdown("""
    <div class="predict-box">
        <h4 style="color:#1565C0;">🏠 Enter house details to predict the median house value in Indian Rupees (&#8377;)!</h4>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    feat_df   = pd.DataFrame(housing.data, columns=housing.feature_names)
    feat_mins = feat_df.min(); feat_maxs = feat_df.max(); feat_means = feat_df.mean()

    feature_descriptions = {
        "MedInc":    "Median Income (in $10,000s)",
        "HouseAge":  "House Age (years)",
        "AveRooms":  "Avg Rooms per Household",
        "AveBedrms": "Avg Bedrooms per Household",
        "Population":"Block Population",
        "AveOccup":  "Avg Occupants per Household",
        "Latitude":  "Latitude",
        "Longitude": "Longitude"
    }

    st.subheader("📥 Input House Feature Values")
    cols_pairs = [st.columns(2) for _ in range(4)]
    flat_cols  = [c for pair in cols_pairs for c in pair]
    user_reg_inputs = {}
    for i, feat in enumerate(housing.feature_names):
        with flat_cols[i]:
            user_reg_inputs[feat] = st.number_input(
                feature_descriptions.get(feat, feat),
                min_value=float(feat_mins[feat]), max_value=float(feat_maxs[feat]),
                value=float(round(feat_means[feat],2)),
                step=0.01 if feat_maxs[feat]<10 else 1.0,
                key=f"reg_input_{feat}"
            )

    st.write("")

    if st.button("🏠 Predict House Price", key="reg_predict_btn"):

        user_arr    = np.array([[user_reg_inputs[f] for f in housing.feature_names]])
        user_scaled = scaler.transform(user_arr)
        reg_pred    = best_regressor.predict(user_scaled)[0]

        price_usd = reg_pred * 100_000          # dataset unit = $100k
        price_inr = price_usd * USD_TO_INR      # convert to INR

        inr_short = format_inr_symbol(price_inr)          # ₹X.XX Cr / L
        inr_full  = f"&#8377;{price_inr:,.0f}"             # ₹12,34,56,789
        usd_fmt   = f"${price_usd:,.0f}"

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        # ── Big Dark Card ─────────────────────────────────
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">&#127968; Predicted Median House Value</div>
            <div class="result-price-inr">{inr_short}</div>
            <div class="result-price-full">{inr_full}</div>
            <div class="result-price-sub">
                &#127482;&#127480; Equivalent: <strong>{usd_fmt}</strong>
                &nbsp;&nbsp;|&nbsp;&nbsp; Exchange Rate: 1 USD = &#8377;{USD_TO_INR}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # ── Three Metric Cards ────────────────────────────
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Raw Output</div>
                <div class="metric-value">{reg_pred:.4f}</div>
                <div style="color:#90A4AE;font-size:12px;">&#215; $100,000 units</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:#FF9800;">
                <div class="metric-label">Price in USD</div>
                <div class="metric-value" style="color:#E65100;">{usd_fmt}</div>
                <div style="color:#90A4AE;font-size:12px;">US Dollars</div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            lakh_val = price_inr / 1_00_000
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:#4CAF50;">
                <div class="metric-label">Price in INR</div>
                <div class="metric-value" style="color:#2E7D32;">{inr_short}</div>
                <div style="color:#90A4AE;font-size:12px;">= &#8377;{lakh_val:,.2f} Lakhs</div>
            </div>""", unsafe_allow_html=True)

        st.write("")

        # ── Price Comparison Bar Chart ────────────────────
        st.subheader("💰 Price Breakdown — USD vs INR Visual")
        fig_p, ax_p = plt.subplots(figsize=(9,5))
        fig_p.patch.set_facecolor("#f5f7fa"); ax_p.set_facecolor("#f5f7fa")

        # Show INR in Lakhs so both bars are comparable
        usd_val_plot = price_usd
        inr_val_plot = price_inr / 1_00_000   # in Lakhs

        categories = [f"USD Price\n({usd_fmt})", f"INR Price\n({inr_short})"]
        values     = [usd_val_plot, inr_val_plot]
        colors     = ["#1565C0", "#FF8F00"]

        bars = ax_p.bar(categories, values, color=colors, width=0.45,
                        edgecolor="white", linewidth=2, zorder=3)
        ax_p.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax_p.set_axisbelow(True)

        labels = [usd_fmt, f"{inr_short}\n(Rs.{inr_val_plot:,.2f} L)"]
        for bar, lbl, color in zip(bars, labels, colors):
            ax_p.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + max(values)*0.025,
                      lbl, ha="center", va="bottom",
                      fontsize=12, fontweight="bold", color=color)

        ax_p.set_ylabel("Value (USD  |  INR in Lakhs)", fontsize=11)
        ax_p.set_title("Predicted House Price — USD vs INR", fontsize=13, fontweight="bold", pad=14)
        ax_p.spines[["top","right"]].set_visible(False)
        ax_p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        st.pyplot(fig_p)

        # ── Input Summary ─────────────────────────────────
        st.subheader("📋 Your Input Summary")
        st.table(pd.DataFrame({
            "Feature": [feature_descriptions.get(f,f) for f in housing.feature_names],
            "Value Entered": [user_reg_inputs[f] for f in housing.feature_names]
        }))

        # ── Your Input vs Dataset Average ─────────────────
        st.subheader("📈 Your Input vs Dataset Average")
        cmp_df = pd.DataFrame([{
            "Feature": feature_descriptions.get(feat,feat),
            "Your Value": round(user_reg_inputs[feat],3),
            "Dataset Average": round(float(feat_means[feat]),3)
        } for feat in housing.feature_names])
        st.dataframe(cmp_df, use_container_width=True)

        fig_bar, ax_bar = plt.subplots(figsize=(12,5))
        x=np.arange(len(housing.feature_names)); w=0.35
        ax_bar.bar(x-w/2,[user_reg_inputs[f] for f in housing.feature_names],w,label="Your Input",color="#42A5F5",edgecolor="black")
        ax_bar.bar(x+w/2,[float(feat_means[f]) for f in housing.feature_names],w,label="Dataset Avg",color="#FFA726",edgecolor="black")
        ax_bar.set_xticks(x); ax_bar.set_xticklabels(housing.feature_names,rotation=30,ha="right")
        ax_bar.set_title("Your Input vs Dataset Average"); ax_bar.legend()
        st.pyplot(fig_bar)

        st.success(f"✅ Predicted Median House Value: **{inr_short}** ({inr_full})")
        st.info(f"ℹ️ Conversion: {reg_pred:.4f} × $100,000 = {usd_fmt} × ₹{USD_TO_INR} = {inr_short}")

# =========================================================
# FORMULA PAGE
# =========================================================

elif menu == "📐 Formula Page":

    st.title("📐 Important Formulas")
    st.markdown("---")

    for title, latex in [
        ("Entropy",          r"Entropy(S) = -\sum p(x)\log_2 p(x)"),
        ("Information Gain", r"IG = Entropy(parent) - Weighted\ Entropy(children)"),
        ("Gini Index",       r"Gini = 1 - \sum p(x)^2"),
        ("Accuracy",         r"Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}"),
        ("MAE",              r"MAE = \frac{1}{n}\sum |y - \hat{y}|"),
        ("MSE",              r"MSE = \frac{1}{n}\sum (y - \hat{y})^2"),
        ("RMSE",             r"RMSE = \sqrt{MSE}"),
        ("R² Score",         r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}}"),
    ]:
        st.header(title)
        st.latex(latex)
        st.markdown("---")

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
