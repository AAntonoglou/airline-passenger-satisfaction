# Airline Passenger Satisfaction Prediction: Machine Learning vs. Deep Learning

This repository contains the complete workflow, experimental analysis, and comparison between traditional Machine Learning models and Deep Learning architectures (DNN) for predicting airline passenger satisfaction.

### 📂 Two-Part Project Structure
This project was developed in **two distinct phases** to evaluate and compare progressive approaches:
1. **Part 1 (Machine Learning - TP6):** Focuses on traditional classifiers (Logistic Regression, KNN, Decision Tree, SVM, Random Forest, and Gradient Boosting). Hyperparameter tuning, computational cost, and business-focused metrics (F1-Score and Recall) were analyzed.
2. **Part 2 (Deep Learning - TP8):** Adds Deep Neural Networks (DNN) under different configurations (base, unregularized, and regularized with L2, Batch Normalization, and Dropout), comparing them directly against the traditional models using statistical robustness tests (McNemar & Bootstrapping).

To reflect this modular progress, **the repository includes 2 separate Notebooks and 2 comprehensive PDF Reports** containing the final business and technical conclusions of each stage.

---

## 📋 Project Overview & Business Goal
The goal is to solve a **binary classification** problem: predicting whether a passenger will end up **dissatisfied** (`neutral or dissatisfied`, encoded as the positive class `1`) or **satisfied** (`satisfied`, class `0`), using demographic data, flight characteristics, and in-flight service ratings.

### Business Approach (Positive Class & Cost-Sensitivity)
- **Positive Class (1 = Dissatisfied):** Missing a dissatisfied passenger (False Negative) is much more expensive than a False Positive (offering retention perks unnecessarily). Thus, **Recall** is our tie-breaker metric.
- **Primary Metric:** **F1-score** on the positive class to balance Precision and Recall, avoiding bias toward the majority class (56.7% of the dataset).

---

## 🛠️ Technologies & Environment
- **Languages & Frameworks:** Python, `Scikit-Learn`, `TensorFlow` {tf_ver}, `Keras` 3
- **Data Wrangling & Plots:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Reproducibility:** Experiments are limited to a **single CPU core** (`n_jobs=1` and single-threaded TensorFlow configuration) to ensure fair computational cost comparisons.

---

## 📊 Final Model Comparison Table
Below are the consolidated results sorted by **F1-score** on the test set (comprising 25,893 records):

| Model | Accuracy | Precision | Recall | F1-score | Training Time (s) | Inference Time (s) | FN (Lower = Better) | FP (Lower = Better) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DNN compleja (L2+BN+Dropout)** | **0.9648** | 0.9567 | **0.9818** | **0.9691** | 101.15 | 0.35 | **265** | 646 |
| **Random Forest** | 0.9632 | 0.9565 | 0.9789 | 0.9676 | 26.98 | 1.07 | 307 | 646 |
| **Gradient Boosting** | 0.9625 | 0.9559 | 0.9782 | 0.9669 | 48.87 | 0.14 | 317 | 655 |
| **DNN base (Quick)** | 0.9568 | 0.9507 | 0.9736 | 0.9620 | 62.02 | 0.16 | 384 | 734 |
| **DNN sin regularizacion** | 0.9570 | 0.9561 | 0.9678 | 0.9619 | 18.73 | 0.17 | 468 | **646** |
| **SVM (RBF)** | 0.9550 | 0.9517 | 0.9690 | 0.9603 | 23.05 | 8.19 | 450 | 714 |
| **Decision Tree** | 0.9517 | 0.9520 | 0.9625 | 0.9572 | 0.47 | **0.01** | 545 | 705 |
| **KNN** | 0.9277 | 0.9136 | 0.9621 | 0.9372 | **0.02** | 4.53 | 550 | 1322 |
| **Logistic Regression** | 0.8718 | 0.8736 | 0.9021 | 0.8876 | 0.28 | **0.01** | 1423 | 1897 |

*Note: SVM was trained on a representative sample of 40k rows due to super-linear $O(n^2)$ scaling constraint.*

---

## 💡 Key Takeaways

1. 🏆 **Deep Learning vs. ML Winner:** The **{mejor_dnn}** achieves the absolute best F1-score (0.9691) and the lowest False Negative rate (only 265 missed dissatisfied passengers). McNemar paired tests and 95% Bootstrap intervals confirm this improvement is statistically significant over the ensembles, changing our previous TP6 recommendation (Gradient Boosting).
2. ⚖️ **Capacity requires Regularization:** The deep unregularized network quickly overfitted (val_loss minimum at epoch 12). Adding L2 regularization, Batch Normalization, and Dropout unlocked the model's full potential, lifting F1-score by 0.0072.
3. ⏱️ **Inference Latency Trade-offs:**
   - **KNN** trains instantly but is extremely slow at query time (4.53s), making it unsuitable for live production environments.
   - **Gradient Boosting** remains an excellent lightweight champion: it predicts 2.5 times faster than the complex DNN with a loss of only 0.002 in F1-score.
4. 📈 **Feature Importance:** Feature analysis indicates that **Online boarding** and **Inflight wifi service** are the highest business-leverage variables to influence satisfaction.

---

## 📂 Repository Structure
- `Notebooks/`
  - `TP6_Machine_Learning.ipynb` - Part 1: Traditional Machine Learning experimental pipeline.
  - `TP8_Deep_Learning.ipynb` - Part 2: Deep Learning evaluation, comparison, and robustness test workflow.
- `Informes_Conclusiones/`
  - `Informe_TP6_Machine_Learning.pdf` - Report containing business and technical conclusions of the ML models.
  - `Informe_TP8_Deep_Learning.pdf` - Report containing business and technical conclusions comparing ML vs. DL.
- `resultados_modelos_final.csv` - Dataframe with the final metrics of all 9 models.
- `figuras/` - Directory containing the plots generated during execution (distributions, learning curves, tradeoffs, and errors).
