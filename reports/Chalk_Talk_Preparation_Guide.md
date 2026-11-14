# Complete Chalk Talk & Viva Preparation Guide
### DS605: Fundamentals of Machine Learning — Course Project Defense
**Instructor:** Dr. Arpit Rana | **Target Presentation:** Blackboard / Whiteboard Defense ("Chalk Talk")

---

## Table of Contents
1. [The 60-Second Elevator Pitch (Start Confidently)](#1-the-60-second-elevator-pitch)
2. [Astronomy Foundations Explained in Simple Words](#2-astronomy-foundations-in-simple-words)
3. [The Data Story: Cleaning & Preprocessing (Phases 1–3)](#3-the-data-story-cleaning--preprocessing)
4. [Feature Engineering & Dimensionality Reduction (Phase 4)](#4-feature-engineering--dimensionality-reduction)
5. [The Machine Learning Models & Chalkboard Derivations (Phase 5)](#5-the-machine-learning-models--chalkboard-derivations)
6. [Evaluation Metrics & The Class Imbalance Dilemma](#6-evaluation-metrics--the-class-imbalance-dilemma)
7. [The Redshift Paradox (Why 88.2% is Better Than 97.6%)](#7-the-redshift-paradox)
8. [Top 20 Chalk Talk Questions & Exact Model Answers](#8-top-20-chalk-talk-questions--model-answers)
9. [How to Run and Demo the Streamlit Application](#9-how-to-run-and-demo-the-streamlit-application)

---

## 1. The 60-Second Elevator Pitch

> *"Good afternoon, Professor and TAs. Our project is **Astronomical Object Classification Using Machine Learning** on the **Sloan Digital Sky Survey (SDSS DR17)** catalog.*
>
> *Every night, modern sky telescopes image millions of celestial objects. Our goal is to automatically classify these objects into three physical categories: **Stars, Galaxies, and Quasars**.*
>
> *We took 100,000 real telescope observations across five photometric optical filters: $u, g, r, i, z$. We cleaned sensor artifacts, proved that raw magnitudes are confounded by distance, and engineered domain-specific **Colour Indices** ($u-g, g-r, r-i, i-z, u-r$) that represent the intrinsic temperature and spectral profile of the celestial source.*
>
> *We evaluated seven machine learning hypothesis classes spanning linear baselines, lazy learners, support vector machines, decision trees, and ensembles. Under realistic photometric survey conditions, our champion **Random Forest** model achieves **88.20% Accuracy** and a **Macro F1-Score of 0.8524**, with a multiclass **ROC-AUC of 0.9653**. We also conducted a scientific ablation study on spectroscopic redshift and deployed a live interactive **Streamlit web application** for real-time astronomical inference."*

---

## 2. Astronomy Foundations in Simple Words

### What are we classifying?
1. **Stars (⭐ STAR):** Single balls of burning hydrogen and helium gas inside our own Milky Way galaxy. In a telescope image, they look like sharp, point-like dots. Their light is thermal blackbody radiation governed by their surface temperature.
2. **Galaxies (🌌 GALAXY):** Enormous collections of billions of stars, gas, and dark matter far outside our Milky Way. In a telescope image, they look "fuzzy" or extended because we can resolve their physical disks or elliptical shapes.
3. **Quasars (⚡ QSO — Quasi-Stellar Object):** The ultra-bright centers of newborn, active galaxies billions of light-years away, powered by a supermassive black hole swallowing millions of tons of gas. Because they are so far away, they look like point-like dots (just like stars!), but their physics is completely different.

### What are $u, g, r, i, z$?
Telescopes do not see in RGB color like human eyes. Instead, astronomers place 5 colored glass filters in front of the camera:
- **$u$ (Ultraviolet, ~3543 Å):** Violet/UV light. Quasars are very bright here because accreting black holes emit intense UV light.
- **$g$ (Green, ~4770 Å):** Blue-green visible light.
- **$r$ (Red, ~6231 Å):** Visible red light. Normal stars and galaxies emit strongly here.
- **$i$ (Near-Infrared 1, ~7625 Å):** Just beyond human vision.
- **$z$ (Near-Infrared 2, ~9134 Å):** Deep infrared. Dust penetrates light here.

### The Inverse Magnitude Scale (Crucial Point!)
In astronomy, brightness is measured in **magnitudes ($m$)**, which is an **inverse logarithmic scale**:
$$m = -2.5 \log_{10}\left(\frac{\text{Flux}}{\text{Flux}_0}\right)$$
- **Smaller or more negative magnitude = BRIGHTER object.**
- **Larger magnitude = FAINTER object.**
*(Analogy: Think of ranking 1st place vs 10th place in a race. 1st is faster than 10th. Magnitude 15 is brighter than magnitude 22!)*

---

## 3. The Data Story: Cleaning & Preprocessing

### Why did you drop the row with `-9999`?
- **Finding:** In Phase 1 inspection, exactly one row out of 100,000 had $u = -9999, g = -9999, z = -9999$.
- **Why?** `-9999` is not a real magnitude (that would mean brighter than the Sun!). In astronomical CCD data pipelines, `-9999` is an error code used when a camera pixel is damaged, saturated, or failed during readout.
- **Action:** Since it affected only **1 out of 100,000 rows (0.001%)**, we simply dropped this single row. The final clean dataset contains **99,999 rows**. No imputation was necessary!

### Why keep negative redshifts?
- Some stars had $z = -0.009$. In physics, negative redshift means **Doppler blueshift** (the star is moving toward Earth inside our Milky Way). It is a real physical phenomenon, not corrupted data!

### Why preserve `obj_ID` as a string?
- `obj_ID` is a 19-digit numerical identifier. If loaded as a standard 64-bit float, IEEE floating-point roundoff causes precision loss in the last digits. Storing as a string preserves the exact primary key!

---

## 4. Feature Engineering & Dimensionality Reduction

### Why are raw magnitudes not enough?
A raw magnitude ($r = 18.5$) tells you how bright an object appears to us on Earth. But:
- A dim, small star close to Earth (e.g., 50 light-years away) can look like $r = 18.5$.
- A giant galaxy billions of light-years away can also look like $r = 18.5$.
Apparent magnitude mixes intrinsic power with distance!

### The Magic of Colour Indices ($u-g, g-r, r-i, i-z$)
When you subtract one magnitude from another:
$$\text{Colour Index} = m_1 - m_2 = -2.5 \log_{10}\left(\frac{F_1}{F_2}\right)$$
**The distance term cancels out!**
The colour index measures the **ratio of flux at two different wavelengths**, which is the slope of the object's **Spectral Energy Distribution (SED)**:
- $u - g$: UV excess (identifies hot quasars).
- $g - r$: Optical slope (tells you the surface temperature of a star or whether a galaxy has young blue stars vs old red stars).
- $u - r$: The famous **galaxy bimodality** tracer (separates red elliptical galaxies from blue spiral galaxies).

### Principal Component Analysis (PCA)
- Standardized all 16 photometric features ($z = \frac{x - \mu}{\sigma}$).
- Calculated the sample covariance matrix $\Sigma = \frac{1}{N-1} X^T X$.
- Solved the eigenvalue problem $\Sigma v_i = \lambda_i v_i$.
- **Result:**
  - $\text{PC}_1$: Explains **41.2%** of variance (overall flux level).
  - $\text{PC}_2$: Explains **32.3%** of variance (UV-optical contrast).
  - $\text{PC}_3$: Explains **13.6%** of variance (NIR curvature).
  - **First 3 components capture 87.1% of the total information!**

---

## 5. The Machine Learning Models & Chalkboard Derivations

*During your chalk talk, write these core equations on the board!*

```
                       ┌─────────────────────────┐
                       │  Supervised Classifier  │
                       └────────────┬────────────┘
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
   Linear Models               Tree Ensembles             Lazy Learner
  - Logistic Reg. (Softmax)   - Decision Tree (CART)     - k-NN (Minkowski)
  - Linear SVM (Hinge Loss)   - Random Forest (Bagging)
                              - Gradient Boost (Residuals)
```

---

### A. Multinomial Logistic Regression (Softmax)
- **Concept:** Eager parametric linear classifier.
- **Hypothesis Function:** Computes linear score $z_k = w_k^T x + b_k$ for each class $k \in \{1, 2, 3\}$, then normalizes to probabilities using Softmax:
  $$P(Y = k | x) = \frac{e^{w_k^T x + b_k}}{\sum_{j=1}^K e^{w_j^T x + b_j}}$$
- **Loss Function:** Categorical Cross-Entropy Loss with $L_2$ regularization:
  $$\mathcal{L}(W) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log P(Y = k | x_i) + \frac{\lambda}{2} \sum_{k=1}^K \|w_k\|^2$$
- **Chalkboard Insight:** Linear decision boundaries form flat hyperplanes. Because astronomical color spaces have curved overlapping clusters, Logistic Regression only achieves **67.86% accuracy**.

---

### B. Support Vector Machines (Linear SVM & Hinge Loss)
*(Explicitly mentioned in Dr. Arpit Rana's syllabus: "Classification using Hinge Loss")*
- **Concept:** Find the maximum-margin hyperplane that separates classes with the widest margin $\frac{2}{\|w\|}$.
- **Primal Optimization Problem (Soft Margin with Slack $\xi_i$):**
  $$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{subject to} \quad y_i(w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
- **Hinge Loss Formulation:**
  $$\ell(y, f(x)) = \max(0, 1 - y \cdot f(x))$$
  - If $y \cdot f(x) \ge 1$: The point is correctly classified outside the margin $\implies \text{Loss} = 0$.
  - If $y \cdot f(x) < 1$: The point violates the margin $\implies \text{Loss} = 1 - y \cdot f(x)$.
- **What is $C$?**
  - High $C$: Heavy penalty for violations $\implies$ narrow margin (risk of overfitting).
  - Low $C$: Tolerates more violations $\implies$ wider margin (higher bias, more robust).
- **Result:** Linear SVM achieved **73.89% accuracy**.

---

### C. Decision Tree (CART Algorithm & Gini Impurity)
- **Concept:** Non-parametric hierarchical splitting of feature space into axis-parallel rectangles.
- **Impurity Metric (Gini Impurity):**
  $$I_G(t) = 1 - \sum_{k=1}^K p_k^2$$
  *(If a node is 100% pure, $p_1 = 1 \implies I_G = 1 - 1^2 = 0$.)*
- **Splitting Rule:** Find the feature $j$ and split threshold $\theta$ that maximizes **Information Gain / Impurity Reduction**:
  $$\Delta I(t) = I_G(t) - \frac{N_L}{N} I_G(t_L) - \frac{N_R}{N} I_G(t_R)$$
- **Regularization:** Unconstrained trees grow until pure (high variance / memorization). We set `max_depth=12` to prune and prevent overfitting.
- **Result:** Decision Tree achieved **85.80% accuracy** and **0.8257 Macro F1**.

---

### D. Random Forest (Champion Architecture — 88.20% Accuracy!)
*(Syllabus Unit: "Ensemble Techniques: Bagging and Pasting, Random Forests")*
- **The Core Idea:** An individual decision tree has low bias but high variance. Random Forest combines $B = 100$ trees to dramatically reduce variance without increasing bias.
- **Two Levels of Randomness:**
  1. **Bootstrap Aggregation (Bagging):** Each tree is trained on a random sample of $N$ rows chosen **with replacement**. About $63.2\%$ of rows are selected; the remaining $36.8\%$ form the **Out-Of-Bag (OOB)** validation set!
  2. **Random Feature Subspaces:** At every split, the tree is only allowed to inspect a random subset of $m = \sqrt{p} = \sqrt{16} = 4$ features.
- **The Variance Reduction Formula (Write this on the board!):**
  $$\text{Var}(\bar{T}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
  - $\sigma^2$ is the variance of a single tree.
  - $B$ is the number of trees (100).
  - $\rho$ is the correlation between trees.
  - **By forcing trees to split on random feature subsets, $\rho$ is forced down! As $B \to \infty$, the second term goes to 0, driving overall error down!**
- **Result:** **88.20% Accuracy, 0.8524 Macro F1, 0.9653 ROC-AUC.**

---

### E. Histogram-Based Gradient Boosting (Sequential Learning)
- **Concept:** Unlike Random Forest (where trees are trained in parallel and averaged), Gradient Boosting trains trees **sequentially**.
- **Principle:** Each new tree $h_m(x)$ is trained to predict the **negative gradient (pseudo-residuals)** of the loss function made by the previous ensemble $F_{m-1}(x)$:
  $$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$
  $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
  where $\eta = 0.1$ is the shrinkage (learning rate) that controls step size.
- **Result:** **87.74% Accuracy, 0.8481 Macro F1.** Extremely fast training (11.6 seconds).

---

### F. k-Nearest Neighbors (Lazy Learner)
- **Concept:** Lazy learner ($O(1)$ training time). Keeps all data in memory and computes Minkowski / Euclidean distances at query time.
- **Prediction:** Uses inverse-distance weighting with $k = 15$.
- **Result:** **86.74% Accuracy, 0.8299 Macro F1.**

---

## 6. Evaluation Metrics & The Class Imbalance Dilemma

### The Majority Class Baseline Paradox
Our dataset has:
- GALAXY: 59,445 (**59.45%**)
- STAR: 21,593 (**21.59%**)
- QSO: 18,961 (**18.96%**)

If a model predicts `GALAXY` for every single celestial object:
- **Accuracy = 59.45%!** (Sounds okay to a beginner, but it's a completely useless model).
- **Macro Recall = 33.3%!**
- **Macro F1 = 0.248!**
- Stars detected = **0%**. Quasars detected = **0%**.

### How We Handled Class Imbalance:
1. **Cost-Sensitive Loss Weighting (`class_weight='balanced'`):**
   $$w_k = \frac{N_{\text{total}}}{K \cdot N_k}$$
   - Weight for Galaxy: $100000 / (3 \times 59445) \approx \mathbf{0.56}$
   - Weight for Star: $100000 / (3 \times 21593) \approx \mathbf{1.54}$
   - Weight for Quasar: $100000 / (3 \times 18961) \approx \mathbf{1.76}$
   Errors on minority Quasars are penalized **3.14 times more heavily** than errors on majority Galaxies!
2. **Evaluation via Macro F1:**
   $$\text{Macro F1} = \frac{F1_{\text{Galaxy}} + F1_{\text{Star}} + F1_{\text{QSO}}}{3}$$
   Gives equal weight to all three classes, preventing majority-class dominance.

---

## 7. The Redshift Paradox

### Why did Random Forest jump from 88.2% to 97.6% with Redshift?
In astronomy:
- **Stars** are inside our galaxy ($z \approx 0$).
- **Galaxies** are in the nearby/moderate universe ($0.01 < z < 0.8$).
- **Quasars** are in the deep, early universe ($z > 1.0$).
Redshift acts almost like an exact label!

### The Professor's Trap Question:
*"If adding redshift gives 97.6% accuracy, why did you build a model without it?"*

### Your Bulletproof Answer:
> *"Because in observational astronomy, **photometry and spectroscopy are completely different observational regimes**:*
> 1. *Photometry takes a quick wide-field camera picture of millions of objects in seconds through colored filters.*
> 2. *Spectroscopy requires taking a spectrograph fiber, pointing it at a single object, and collecting light for hours to disperse it and measure Doppler shift.*
> 3. *Spectroscopy costs **100 to 1,000 times more telescope time and money**.*
> 4. *Telescopes like the Rubin Observatory LSST will discover 20 billion objects via photometry. We can never take spectra of all 20 billion! We must classify them using **photometry alone**.*
> 5. *Our 88.2% photometry-only model is the real-world operational model. We ran the redshift experiment strictly as an **ablation study** to verify the physical distribution of redshifts."*

*(Any professor hearing this will give you top marks on the spot!)*

---

## 8. Top 20 Chalk Talk Questions & Model Answers

#### Q1: What is the objective of this project?
**Ans:** To classify celestial objects into Stars, Galaxies, and Quasars based on multi-band photometric survey measurements ($u, g, r, i, z$) from SDSS Data Release 17.

#### Q2: What is the bias-variance tradeoff in our models?
**Ans:** 
- Logistic Regression has **high bias** (underfitting, assumes a flat linear boundary, 67.9% accuracy).
- A single deep Decision Tree has **high variance** (overfitting to training noise).
- Random Forest achieves the sweet spot: by averaging 100 de-correlated trees, it retains the low bias of deep trees while shrinking the variance by $\frac{1-\rho}{B}\sigma^2$, achieving 88.20% accuracy on held-out test data.

#### Q3: Why did you scale features using `StandardScaler`?
**Ans:** Algorithms that rely on distances or gradient steps (Logistic Regression, k-NN, Linear SVM) are sensitive to feature scales. If one filter had magnitude 10 and another had magnitude 30, the larger feature would artificially dominate distance calculations. `StandardScaler` standardizes each feature to mean 0 and unit variance: $z = \frac{x - \mu}{\sigma}$.

#### Q4: How did you prevent data leakage during scaling?
**Ans:** We wrapped `StandardScaler` inside a scikit-learn `Pipeline`. The scaler is fitted strictly on the **training fold** during cross-validation, and never on the test fold or the full dataset.

#### Q5: What is the difference between Bagging and Boosting?
**Ans:**
- **Bagging (Random Forest):** Trees are trained in **parallel** on independent bootstrap resamples. Trees do not communicate. Their predictions are averaged to reduce variance.
- **Boosting (Gradient Boosting):** Trees are trained **sequentially**. Each new tree learns to correct the errors (pseudo-residuals) made by previous trees, reducing bias.

#### Q6: Why did Linear SVM struggle on Stars ($F1 = 0.46$)?
**Ans:** In $(u-g)$ vs $(g-r)$ color space, stars form a curved "stellar locus" with high curvature. A linear hyperplane cannot slice a curved manifold without misclassifying large sections into galaxies or quasars. Non-linear tree ensembles handle this effortlessly.

#### Q7: What is Gini Impurity, and how is it different from Entropy?
**Ans:**
- Gini Impurity: $I_G = 1 - \sum p_i^2$ (computationally faster, no logarithms).
- Entropy: $H = -\sum p_i \log_2 p_i$ (information theory measure).
Both reach 0 for completely pure nodes and maximize at uniform distributions. In practice, both give nearly identical split quality.

#### Q8: What is the Out-Of-Bag (OOB) error in Random Forest?
**Ans:** When bootstrapping with replacement, the probability of a row not being picked is $(1 - \frac{1}{N})^N \approx e^{-1} \approx 36.8\%$. These unpicked rows serve as a built-in cross-validation set (OOB sample) for that tree, providing an unbiased error estimate without a separate validation split.

#### Q9: What is the One-vs-Rest (OvR) strategy?
**Ans:** For $K = 3$ classes, OvR trains 3 separate binary classifiers:
1. Galaxy vs (Star + QSO)
2. Star vs (Galaxy + QSO)
3. QSO vs (Galaxy + Star)
The class with the highest confidence or probability is selected as the final prediction.

#### Q10: What is the difference between Macro F1 and Weighted F1?
**Ans:**
- **Macro F1:** Unweighted arithmetic mean: $\frac{F1_1 + F1_2 + F1_3}{3}$. Treats minority Quasars as equally important as majority Galaxies.
- **Weighted F1:** Weighted by class frequencies: $w_1 F1_1 + w_2 F1_2 + w_3 F1_3$. Can mask poor minority performance.

#### Q11: What is the main source of misclassification in our best model?
**Ans:** Confusion between **Stars and Quasars**. Both appear as point sources in optical imaging without resolved spatial disks, and hot blue stars overlap with the photometric colors of moderate-redshift quasars.

#### Q12: What does an ROC curve measure?
**Ans:** It plots True Positive Rate (Sensitivity) vs False Positive Rate ($1 - \text{Specificity}$) across all classification thresholds. The Area Under the Curve (AUC) measures the probability that the classifier ranks a randomly chosen positive instance higher than a randomly chosen negative instance. Our Random Forest achieved an outstanding **0.9653 Macro AUC**.

#### Q13: Why did you use `HistGradientBoostingClassifier` instead of standard `GradientBoostingClassifier`?
**Ans:** Standard Gradient Boosting sorts every continuous feature at every split ($O(N \log N)$), which is slow on 80,000 samples. Histogram-based boosting discretizes continuous features into 256 integer bins, reducing split evaluation complexity to $O(N)$ and accelerating training by 10x with zero accuracy loss.

#### Q14: How does SMOTE work?
**Ans:** Synthetic Minority Over-sampling Technique (Chawla et al., 2002). For each minority instance $x$:
1. Finds its $k$ nearest neighbors of the same class.
2. Randomly chooses a neighbor $x_{\text{nn}}$.
3. Creates a synthetic point by linear interpolation:
   $$x_{\text{new}} = x + \lambda (x_{\text{nn}} - x) \quad \text{where } \lambda \sim U(0, 1)$$

#### Q15: Why didn't you drop correlated bands like `r` and `i` ($r = 0.97$)?
**Ans:** Collinearity in features affects the interpretability of individual linear regression coefficients, but does not harm tree-based ensembles. More importantly, subtracting adjacent correlated bands produces our most informative features: the **colour indices**!

#### Q16: How did you select hyperparameters?
**Ans:** Using Stratified 5-Fold Cross Validation. For Random Forest, we evaluated `n_estimators` (50, 100, 150) and `max_depth` (10, 15, None). 100 trees with `max_depth=15` provided the optimal balance of generalization without overfitting.

#### Q17: What is Pogson's ratio?
**Ans:** A difference of 5 astronomical magnitudes corresponds to a factor of exactly 100 in observed brightness flux: $\sqrt[5]{100} \approx 2.512$.

#### Q18: What is the Voting Classifier?
**Ans:** An ensemble of ensembles. Our Soft Voting classifier takes the predicted probability distributions of Random Forest, Gradient Boosting, and Logistic Regression, averages them, and outputs the class with the maximum average probability. It achieved 87.71% accuracy.

#### Q19: What is the biggest practical limitation of this dataset?
**Ans:** The SDSS17 CSV represents a catalog of objects that were selected for spectroscopic follow-up. This introduces a "spectroscopic target selection bias," meaning the catalog distribution may not match the raw, unfiltered sky footprint of faint objects.

#### Q20: How would you deploy this model in an astronomical observatory?
**Ans:** As demonstrated in our Streamlit application (`app.py`), the model pipeline is saved as a `.joblib` serialized file. When the observatory camera completes an optical exposure, raw calibrated fluxes ($u, g, r, i, z$) are streamed into the pipeline, engineered into color indices, and classified in under 5 milliseconds per object.

---

## 9. How to Run and Demo the Streamlit Application

To demonstrate your project to the examiner or professor:

```powershell
# Open terminal in the project directory
streamlit run app.py
```
*(Or if running directly with python: `python -m streamlit run app.py`)*

### Recommended 3-Minute Live Demo Flow:
1. **Show the Live Classifier Tab:**
   - Click "Preset: Main-Sequence Star" $\to$ Model predicts **STAR** with 94% confidence.
   - Click "Preset: Luminous Red Galaxy" $\to$ Model instantly switches to **GALAXY** with 98% confidence.
   - Click "Preset: High-Redshift Quasar" $\to$ Model switches to **QSO**.
   - Point out the **2D Colour-Colour Diagram**: show how the query object's red diamond marker moves across the real SDSS empirical background!
2. **Show the Batch CSV Tab:**
   - Click "Download Sample Test CSV".
   - Upload the sample CSV into the file uploader.
   - Show the instant batch classification, the predicted donut chart, and the download button.
3. **Show the Model Benchmark Tab:**
   - Display the comparison bar chart and normalized confusion matrices.
   - Point to the **Redshift Ablation Chart** and explain the observational cost difference!
4. **Show the Chalk Talk Tab:**
   - Show that all mathematical formulations and theoretical justifications from Dr. Arpit Rana's syllabus are embedded right inside the application!
