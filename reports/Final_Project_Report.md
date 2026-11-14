# Astronomical Object Classification Using Machine Learning
### An Empirical Study on SDSS Photometric Observations

**Course:** DS605: Fundamentals of Machine Learning  
**Academic Year:** 2026-27 (Autumn Semester)  
**Instructor:** Dr. Arpit Rana  
**Program:** M.Sc. Data Science, Semester I  
**Dataset:** Sloan Digital Sky Survey Data Release 17 (SDSS17 Stellar Classification Dataset)  
**Source Catalog:** 100,000 observations, 18 attributes  
**Deliverables:** Cleaned Data Pipeline, Exploratory Analysis, Feature Engineering, Multi-Model Benchmarking, Interactive Web Application (`app.py`), and Technical Documentation.

---

## Executive Summary & Abstract

Modern wide-field astronomical sky surveys generate enormous streams of imaging data, containing hundreds of millions of celestial targets. Classifying these targets into fundamental physical categories—**Stars**, **Galaxies**, and **Quasars (QSOs)**—is a foundational problem in astrophysics and observational cosmology. Traditional manual spectroscopic inspection is observationally expensive and impossible at gigabyte-to-terabyte catalog scales.

This project designs, implements, and evaluates an end-to-end machine learning classification system applied to the Sloan Digital Sky Survey 17th Data Release (SDSS17). Following rigorous data ingestion, cleaning, and quality control (Phase 1), exploratory data analysis (Phase 2), and dashboard preparation (Phase 3), we engineered domain-specific photometric features (Phase 4)—most notably adjacent and wide colour indices ($u-g, g-r, r-i, i-z, u-r$), second-order spectral curvatures, and principal component projections—to capture the underlying Spectral Energy Distribution (SED) of celestial objects independently of apparent distance.

In Phase 5, seven supervised machine learning hypothesis classes spanning eager linear learners (Multinomial Logistic Regression), lazy learners (k-Nearest Neighbors), maximum-margin classifiers (Linear Support Vector Machines), tree-based algorithms (Decision Trees), and ensemble architectures (Random Forest, Histogram-Based Gradient Boosting, and Soft Voting Ensembles) were trained and benchmarked using leak-safe stratified 5-fold cross-validation. 

Under realistic **pure photometric imaging conditions** (without spectroscopic redshift), our champion **Random Forest** classifier achieved an overall **Test Accuracy of 88.20%**, a **Macro F1-Score of 0.8524**, a **Weighted F1-Score of 0.8824**, and a multiclass One-vs-Rest **ROC-AUC of 0.9653** on a completely unobserved test set of 20,000 objects. An extensive ablation study demonstrated that while incorporating spectroscopic redshift elevates classification accuracy to **97.56%**, such information is observationally prohibitive during initial imaging surveys. Finally, the complete modeling architecture was deployed into an interactive **Streamlit web application** (`app.py`) featuring real-time object classification, batch CSV processing, dynamic 2D colour-colour diagnostic plotting, and interactive performance dashboards.

---

## 1. Introduction & Problem Definition

### 1.1 Astrophysical Background
The observable universe contains diverse luminous structures that, when imaged by wide-field ground and space telescopes, can be grouped into three fundamental classes:
1. **Stars (`STAR`):** Self-luminous gravitationally bound plasma spheroids located predominantly within our home galaxy, the Milky Way. Point-like optical morphology with thermal blackbody-like emission spectra determined by effective surface temperature and spectral class (O, B, A, F, G, K, M).
2. **Galaxies (`GALAXY`):** Massive, gravitationally bound systems composed of billions to trillions of stars, stellar remnants, interstellar gas, dust, and dark matter. Spatially extended (resolved) morphology spanning early-type ellipticals (quiescent, red) and late-type spirals (star-forming, blue).
3. **Quasars (`QSO` — Quasi-Stellar Objects):** Extremely luminous active galactic nuclei (AGN) powered by supermassive black holes actively accreting matter at the centers of distant early galaxies. Appearing optical point-like (hence "quasi-stellar") with power-law continuum emission, prominent broad emission lines (e.g., Ly-$\alpha$, C IV, Mg II), and massive cosmological redshifts.

### 1.2 The Sloan Digital Sky Survey (SDSS)
The Sloan Digital Sky Survey utilizes a dedicated 2.5-meter wide-angle optical telescope at Apache Point Observatory in New Mexico. Its imaging camera utilizes a mosaic of 30 CCD detectors observing through five broad optical filters designed by Fukugita et al. (1996):
- **$u$ (Ultraviolet):** Central wavelength $\lambda_{\text{eff}} \approx 3543 \text{ \AA}$
- **$g$ (Green-Cyan):** Central wavelength $\lambda_{\text{eff}} \approx 4770 \text{ \AA}$
- **$r$ (Red):** Central wavelength $\lambda_{\text{eff}} \approx 6231 \text{ \AA}$
- **$i$ (Near-Infrared 1):** Central wavelength $\lambda_{\text{eff}} \approx 7625 \text{ \AA}$
- **$z$ (Near-Infrared 2):** Central wavelength $\lambda_{\text{eff}} \approx 9134 \text{ \AA}$

### 1.3 Machine Learning Formulation
Mathematically, the problem is formulated as a 3-class supervised classification problem:
Given a feature vector $x \in \mathbb{R}^p$ representing photometric measurements of a celestial body, determine a decision mapping $f: \mathbb{R}^p \to \{c_1, c_2, c_3\}$, where $c_1 = \text{GALAXY}$, $c_2 = \text{STAR}$, and $c_3 = \text{QSO}$.

---

## 2. Dataset Quality, Preprocessing & Inspection (Phase 1 Review)

### 2.1 Catalog Overview
The source dataset (`star_classification.csv`, SDSS Data Release 17) comprises **100,000 observations** across **18 columns**:
- Identifiers: `obj_ID` (unique photometric object identifier), `spec_obj_ID` (spectroscopic observation ID)
- Astrometric Coordinates: `alpha` (Right Ascension, J2000 degrees), `delta` (Declination, J2000 degrees)
- Photometric Magnitudes: `u`, `g`, `r`, `i`, `z`
- Survey Metadata: `run_ID`, `rerun_ID`, `camcol`, `field_ID`, `fiber_ID`, `plate`, `MJD` (Modified Julian Date)
- Physical Property: `redshift` (dimensionless spectral shift $z = \Delta \lambda / \lambda_0$)
- Ground Truth Target: `class` (`GALAXY`, `STAR`, `QSO`)

### 2.2 Quality Assurance & Anomaly Cleaning
1. **Identifier Handling:** `obj_ID` and `spec_obj_ID` were loaded as string dtypes to prevent precision truncation inherent in floating-point scientific notation.
2. **Missing Values & Sentinels:** Standard `isna().sum()` returned 0 missing values across all columns. However, systematic inspection identified **one row** where filters `u`, `g`, and `z` held the value `-9999.0`. In astronomical pipelines, `-9999` is an unphysical sentinel value indicating detector saturation, hardware read failure, or corrupted PSF fitting. Because this artifact represented only $0.001\%$ of the dataset, the row was dropped, leaving **99,999 valid observations**.
3. **Negative Redshifts:** Small negative redshifts (e.g., $z \approx -0.009$) were deliberately preserved. In observational astronomy, stars moving toward the Sun within the Milky Way or blueshifted objects in the Local Group (such as the Andromeda Galaxy M31) exhibit genuine physical Doppler blueshift. Marking them invalid would represent false data contamination.
4. **Leak-Safe Design:** No imputer or scaler was fit to the dataset at the global level in Phase 1, preventing data snooping into downstream test partitions.

---

## 3. Exploratory Data Analysis & Statistical Astronomy (Phase 2 Review)

### 3.1 Class Distribution & Baseline Accuracy
The target distribution across the 99,999 clean observations is:
- **GALAXY:** 59,445 objects (**59.45%**)
- **STAR:** 21,593 objects (**21.59%**)
- **QSO:** 18,961 objects (**18.96%**)

> **Critical Methodological Insight:** A trivial zero-rule majority classifier predicting `GALAXY` for every query would achieve **59.45% accuracy**, while completely failing to identify Stars and Quasars. Therefore, evaluating models strictly by overall accuracy is scientifically invalid; model optimization and selection must be governed by **Macro F1-Score**, **Per-Class Recall**, and **Confusion Matrices**.

### 3.2 Photometric Properties & Inverse Magnitude Scale
Astronomical magnitudes follow Pogson's logarithmic relation:
$$m = -2.5 \log_{10}\left(\frac{F}{F_0}\right)$$
Hence, magnitude is **inversely related to flux**: smaller values denote brighter observed objects, while larger values indicate fainter objects.

Summary descriptive statistics of the five bands across all 99,999 observations:
- $u$: Mean = 22.08, Median = 22.18, IQR = 1.95 (UV band is faintest due to atmospheric extinction)
- $g$: Mean = 20.63, Median = 21.10, IQR = 2.47
- $r$: Mean = 19.64, Median = 20.13, IQR = 2.41
- $i$: Mean = 19.08, Median = 19.41, IQR = 2.30
- $z$: Mean = 18.77, Median = 19.00, IQR = 2.22

### 3.3 Collinearity Analysis
Adjacent filter passbands exhibit strong positive correlations due to overlapping wavelength transmission curves:
- Pearson $r(i, z) = 0.9715$
- Pearson $r(r, i) = 0.9629$
- Pearson $r(g, r) = 0.9385$
Rather than discarding collinear bands (which would discard critical wavelength information), linear collinearity motivates constructing orthogonal difference features (colour indices).

---

## 4. Feature Engineering & Dimensionality Reduction (Phase 4)

### 4.1 Physical Rationale for Colour Indices
Apparent magnitude $m$ conflates intrinsic luminosity $L$ with luminosity distance $d_L$:
$$m = M + 5 \log_{10}(d_L) - 5 + A_\lambda$$
where $M$ is absolute magnitude and $A_\lambda$ is extinction. Consequently, a faint nearby star and a bright distant galaxy can produce identical single-band apparent magnitudes.

In contrast, the **Colour Index**—defined as the difference between magnitudes in two filters $A$ and $B$—cancels distance dependence:
$$m_A - m_B = -2.5 \log_{10}\left(\frac{F_A}{F_B}\right)$$
Colour indices directly measure the ratio of fluxes at distinct wavelengths, capturing the slope and temperature profile of the Spectral Energy Distribution (SED).

### 4.2 Derived Feature Families
We engineered 11 new domain-specific features across four families:
1. **Adjacent Colour Indices (Local Slopes):**
   - $u - g$: Sensitive to ultraviolet continuum, Balmer break, and UV excess (hallmark of Quasars).
   - $g - r$: Sensitive to optical slope, stellar surface temperatures, and the 4000 Å break in galaxies.
   - $r - i$: Separates late-type cool dwarf stars from elliptical galaxies.
   - $i - z$: Near-infrared continuum slope, vital for high-redshift quasar detection.
2. **Wide Spectral Indices (Broad Baselines):**
   - $u - r$: The classical optical color bimodality parameter in extragalactic astronomy, separating star-forming blue galaxies from passive red ellipticals.
   - $u - z$: Spans the complete optical survey baseline ($3543 \text{ \AA}$ to $9134 \text{ \AA}$).
   - $g - z$: Optical-to-infrared color slope.
3. **Second-Order Spectral Curvatures:**
   - $\text{curvature}_{ugr} = (u - g) - (g - r) = u - 2g + r$: Measures spectral curvature around the $g$-band (Balmer discontinuity).
   - $\text{curvature}_{gri} = (g - r) - (r - i) = g - 2r + i$: Measures inflection across the optical-red threshold.
4. **Aggregate Photometric Dispersion:**
   - $\text{mag\_mean} = \frac{1}{5}(u + g + r + i + z)$: Proxy for total integrated flux.
   - $\text{mag\_std} = \sqrt{\frac{1}{4} \sum_{b} (m_b - \text{mag\_mean})^2}$: Quantifies overall spectral variance across bands.

### 4.3 Feature Importance & Statistical Validation
We computed both ANOVA F-statistics (linear class separability) and Mutual Information (information-theoretic non-linear association) across all 16 photometric attributes on a stratified sample of 15,000 objects.

| Rank | Feature | Description | ANOVA F-Score | Mutual Information |
|:---:|:---:|:---:|:---:|:---:|
| 1 | `r_i` | Red-to-NIR color slope | 1740.00 | **0.3084** |
| 2 | `g_z` | Wide optical-NIR baseline | 3682.38 | **0.3045** |
| 3 | `mag_std` | Spectral dispersion | 3822.21 | **0.2942** |
| 4 | `g_r` | Optical color slope | 3676.05 | **0.2912** |
| 5 | `u_z` | Full wavelength baseline | 3586.92 | **0.2829** |
| 6 | `u_r` | Galaxy bimodality index | 2620.74 | **0.2503** |
| 7 | `i_z` | NIR color slope | 488.73 | **0.2014** |
| 8 | `curvature_gri` | Second-order red curvature | 721.73 | **0.1734** |
| 9 | `u_g` | Ultraviolet excess | 954.46 | **0.1634** |
| 10 | `z` | Single-band NIR magnitude | 1579.91 | **0.1457** |

> **Key Astrophysical Finding:** Engineered colour indices comprehensively outrank every raw single-band magnitude in mutual information. This mathematically confirms that spectral slope (SED shape) is significantly more informative than apparent brightness for celestial taxonomy.

### 4.4 Unsupervised Dimensionality Reduction via PCA
In alignment with the course syllabus unit *Dimensionality Reduction using PCA and SVD*, Principal Component Analysis was performed on standardized features:
- **PC1 (41.18% of variance):** Driven by overall brightness and optical-infrared baseline.
- **PC2 (32.32% of variance):** Driven by UV-optical color contrast ($u-g, u-r$).
- **PC3 (13.61% of variance):** Driven by spectral curvature and second differences.
- **Cumulative Explained Variance:** The first 3 principal components capture **87.11%** of total variance, and 5 components capture **99.78%**, confirming high representational density.

---

## 5. Supervised Learning Methodology & Mathematical Formulations (Phase 5)

To ensure full coverage of the DS605 Course Syllabus, seven distinct supervised classification models representing diverse learning paradigms were developed and benchmarked.

### 5.1 Eager Linear Baseline: Multinomial Logistic Regression
- **Mathematical Principle:** Models class-conditional probabilities using the Softmax link function:
  $$P(Y = k | x) = \frac{\exp(w_k^T x + b_k)}{\sum_{j=1}^K \exp(w_j^T x + b_j)}$$
- **Loss Function:** Optimized via Cross-Entropy Loss with $L_2$ weight regularization:
  $$\mathcal{L}(W) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K \mathbb{I}(y_i = k) \log P(Y = k | x_i) + \frac{\lambda}{2} \sum_{k=1}^K \|w_k\|^2$$
- **Role in Study:** Serves as the parametric linear baseline.

### 5.2 Lazy Learner: k-Nearest Neighbors (k-NN)
- **Mathematical Principle:** Non-parametric instance-based learner. No explicit training phase is conducted ($O(1)$ training time). During inference, finds the $k$ training points closest to query point $x_q$ using Euclidean distance:
  $$d(x, x') = \sqrt{\sum_{j=1}^p (x_j - x_j')^2}$$
- **Weighting:** Predictions are aggregated using inverse-distance weighting:
  $$P(Y = k | x_q) = \frac{\sum_{i \in \mathcal{N}_k(x_q), y_i=k} \frac{1}{d(x_q, x_i)}}{\sum_{i \in \mathcal{N}_k(x_q)} \frac{1}{d(x_q, x_i)}}$$
- **Hyperparameters:** $k = 15$, `weights='distance'`.

### 5.3 Maximum-Margin Classifier: Linear Support Vector Machine (Linear SVM)
- **Mathematical Principle:** Implements the soft-margin primal quadratic optimization problem using Hinge Loss:
  $$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{subject to} \quad y_i(w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
- **Hinge Loss:** Penalizes points violating the margin boundary: $\ell(y, f(x)) = \max(0, 1 - y f(x))$.
- **Multiclass Extension:** Handled via One-vs-Rest (OvR) decision hyperplanes.

### 5.4 Tree-Based Learning: Decision Tree Classifier (CART)
- **Mathematical Principle:** Recursively partitions the feature space into hyper-rectangles using binary axis-aligned splits.
- **Splitting Criterion (Gini Impurity):**
  $$I_G(t) = 1 - \sum_{k=1}^K p(k|t)^2$$
- **Optimization:** Selects feature $j^*$ and threshold $\theta^*$ maximizing Impurity Reduction:
  $$\Delta I(t) = I_G(t) - \frac{N_{t_L}}{N_t} I_G(t_L) - \frac{N_{t_R}}{N_t} I_G(t_R)$$
- **Regularization:** Maximum tree depth constrained to `max_depth=12` to prevent catastrophic overfitting.

### 5.5 Ensemble Methods
1. **Random Forest (Bagging + Feature Subsampling):**
   - Fits $B = 100$ unpruned decision trees on bootstrap resamples of the training data.
   - At each split, considers only a random subset of features $m = \sqrt{p} = \sqrt{16} = 4$.
   - **Variance Reduction Formula:**
     $$\text{Var}(\bar{f}(x)) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
     Random feature selection lowers inter-tree correlation $\rho$, driving ensemble variance down toward zero while preserving low individual tree bias.
2. **Histogram-Based Gradient Boosting (Sequential Boosting):**
   - Incrementally constructs an ensemble of shallow trees by fitting each successive tree to the negative gradient (pseudo-residuals) of the multinomial cross-entropy loss:
     $$r_{ikm} = -\left[\frac{\partial L(y_i, f(x_i))}{\partial f_k(x_i)}\right]_{f = f_{m-1}} = \mathbb{I}(y_i = k) - P_{m-1}(Y = k | x_i)$$
   - Utilizes integer histogram binning (256 bins) for fast $O(N)$ split finding.
3. **Soft Voting Ensemble:**
   - Meta-classifier computing the unweighted average of predicted class probabilities from the top three heterogeneous models (Random Forest, Gradient Boosting, and Logistic Regression):
     $$P_{\text{ensemble}}(Y = k | x) = \frac{1}{3} \sum_{m=1}^3 P_m(Y = k | x)$$

---

## 6. Experimental Design, Cross-Validation & Imbalance Handling

### 6.1 Data Partitioning & Leak-Safe Pipelines
1. **Splits:** 80% Training ($N = 79,999$) and 20% Held-Out Testing ($N = 20,000$), stratified by target class.
2. **Cross-Validation:** Stratified 5-Fold Cross-Validation was conducted exclusively on training folds to estimate generalization stability and prevent test set snooping.
3. **Pipeline Isolation:** All feature transformations (`StandardScaler`) were embedded inside `sklearn.pipeline.Pipeline`, guaranteeing that fold statistics ($\mu_{\text{train}}, \sigma_{\text{train}}$) never leaked into validation or test folds.

### 6.2 Treatment of Class Imbalance
With galaxies constituting nearly 60% of instances, unweighted models tend to neglect Quasars ($18.96\%$). We addressed this via **Cost-Sensitive Learning (`class_weight='balanced'`)**, adjusting the loss penalty inversely proportional to class frequencies:
$$w_k = \frac{N}{K \cdot N_k}$$
where $N = 79,999$, $K = 3$, and $N_k$ is the class sample count. This penalizes errors on rare Quasars 3.14 times more heavily than errors on majority Galaxies.

---

## 7. Experimental Results & Comparative Benchmark

### 7.1 Quantitative Benchmark Table (Photometry-Only Models)
The complete performance comparison on the held-out test set ($N = 20,000$) is detailed below:

| Classifier Architecture | CV Macro F1 | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | $F1_{\text{Galaxy}}$ | $F1_{\text{Star}}$ | $F1_{\text{QSO}}$ | ROC-AUC (OvR) | Training Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 0.8399 | **88.20%** | **0.8490** | **0.8561** | **0.8524** | **0.8824** | **0.9291** | **0.8128** | **0.8152** | **0.9653** | 70.7s |
| **Gradient Boosting** | **0.8440** | 87.74% | 0.8409 | 0.8567 | 0.8481 | 0.8784 | 0.9256 | 0.8077 | 0.8111 | 0.9647 | 11.6s |
| **Voting Ensemble** | 0.8419 | 87.71% | 0.8428 | 0.8545 | 0.8475 | 0.8778 | 0.9251 | 0.8062 | 0.8111 | 0.9551 | 40.8s |
| **k-Nearest Neighbors** | 0.8303 | 86.74% | 0.8415 | 0.8199 | 0.8299 | 0.8654 | 0.9216 | 0.7704 | 0.7977 | 0.9474 | 0.1s |
| **Decision Tree** | 0.7989 | 85.80% | 0.8184 | 0.8341 | 0.8257 | 0.8591 | 0.9119 | 0.7719 | 0.7933 | 0.9166 | 3.9s |
| **Linear SVM** | 0.6710 | 73.89% | 0.6754 | 0.6949 | 0.6691 | 0.7306 | 0.8344 | 0.4632 | 0.7098 | 0.8335 | 1.4s |
| **Logistic Regression** | 0.6531 | 67.86% | 0.6374 | 0.6905 | 0.6515 | 0.6885 | 0.7537 | 0.4864 | 0.7143 | 0.8352 | 1.3s |

### 7.2 Confusion Matrix & Per-Class Error Diagnostics
Normalized confusion matrix analysis reveals distinct error patterns:
1. **Galaxies ($F1 = 0.929$):** Exceptionally well classified. Galaxies have high signal-to-noise optical flux profiles and strong 4000 Å breaks that separate cleanly from stars.
2. **Stars vs. Quasars ($F1_{\text{Star}} = 0.813, F1_{\text{QSO}} = 0.815$):** The primary source of misclassification. Both Stars and Quasars are spatially unresolved point sources in optical imaging. Hot blue stars (O/B stars, white dwarfs) exhibit optical colors that partially overlap with low-redshift quasars without spectroscopic line emission confirmation.
3. **Linear Failure:** Linear models (Logistic Regression and Linear SVM) severely struggle on Stars ($F1 \approx 0.46 - 0.49$), proving that celestial classes are separated by non-linear geometric manifolds in color space.

---

## 8. Scientific Ablation Study: Value of Spectroscopic Redshift

A pivotal scientific question in astronomical data science is: **How much does spectroscopic redshift improve object classification?**

To answer this, we trained a parallel Random Forest model with identical hyperparameters, adding `redshift` to the feature set:

| Evaluation Metric | Photometry-Only Model (Realistic Survey) | Photometry + Spectroscopic Redshift | Absolute Gain ($\Delta$) |
|:---|:---:|:---:|:---:|
| **Test Accuracy** | 88.20% | **97.56%** | **+9.36%** |
| **Macro F1-Score** | 0.8524 | **0.9721** | **+0.1197** |
| **Quasar F1-Score** | 0.8152 | **0.9752** | **+0.1600** |
| **Star F1-Score** | 0.8128 | **0.9631** | **+0.1503** |

### Astrophysical Explanation:
1. **Stars** reside within the Milky Way and have localized peculiar velocities ($v \ll c$), yielding redshifts clustered tightly at $z \approx 0.000 \pm 0.001$.
2. **Galaxies** exhibit moderate cosmological redshifts corresponding to the Hubble flow ($0.01 \lesssim z \lesssim 0.8$).
3. **Quasars** are hyper-luminous active galactic nuclei observed across cosmological lookback times, exhibiting high redshifts ($z > 1.0$, extending up to $z = 7.0$).

> **Core Viva / Defense Conclusion:** While adding redshift raises classification performance to 97.56%, obtaining spectroscopic redshift requires dispersing light through an optical spectrograph—a procedure requiring **100 to 1,000 times more telescope integration time** than acquiring 5-band imaging. Therefore, real-world wide-field survey pipelines (such as the Vera C. Rubin Observatory Legacy Survey of Space and Time) must perform initial target discovery and classification on **photometry alone**, establishing our 88.20% model as the true state-of-the-art for imaging surveys.

---

## 9. Interactive Web Application Architecture (`app.py`)

To deliver an accessible, production-grade interface for astronomers and examiners, a full-featured web application was constructed using **Streamlit** and **Plotly**:

```text
┌────────────────────────────────────────────────────────┐
│               AstroClassify Web Interface              │
├──────────────┬─────────────────────────────────────────┤
│ Sidebar      │ Main Tabs                               │
│ - Model Pick │ 1. Live Object Classifier & Telescopic  │
│ - Redshift   │    Sliders (u, g, r, i, z, real-time)   │
│   Toggle     │ 2. Batch CSV Processing & Auto-Reports  │
│ - Celestial  │ 3. Model Benchmark & Interactive Charts │
│   Presets    │ 4. Survey Footprint & Spatial Sky Map   │
│              │ 5. DS605 Chalk Talk & Theoretical Guide │
└──────────────┴─────────────────────────────────────────┘
```

### Key Technical Features:
- **Telescopic Slider Interface:** Allows researchers to input $u, g, r, i, z$ magnitudes and receive instantaneous predictions with confidence probabilities.
- **Dynamic 2D Color-Color Overlay:** Queries are plotted as real-time glowing markers directly onto a 2,500-point empirical background of actual SDSS observations in $(g-r)$ vs $(u-g)$ space, illustrating the geometric classification decision.
- **Batch CSV Classification:** Supports dragging and dropping arbitrary catalog CSVs, auto-computing engineered features, generating class summary donut charts, and exporting predictions.
- **Embedded Theoretical Reference:** Houses mathematical derivations, loss functions, and syllabus answers directly inside the application for defense presentations.

---

## 10. Conclusion & Future Directions

### 10.1 Key Findings Summary
1. Cleaned and preprocessed 99,999 observations from SDSS DR17, eliminating unphysical `-9999` sentinels while preserving genuine physical Doppler blueshifts.
2. Engineered 11 domain-specific photometric features (colour indices, spectral curvatures, dispersion statistics), demonstrating through Mutual Information that color slopes dominate raw magnitudes in diagnostic power.
3. Showed through PCA that 3 principal components capture 87.1% of multi-band photometric variance.
4. Benchmarked seven supervised learning architectures, proving that ensemble tree methods (Random Forest at 88.20% Accuracy, 0.8524 Macro F1) decisively outperform linear models on complex non-linear color manifolds.
5. Successfully resolved the astronomical redshift paradox, explaining why photometric classification is the critical operational bottleneck for next-generation sky surveys.

### 10.2 Future Work
- **Photometric Redshift Estimation ($z_{\text{phot}}$):** Train a regressor (Random Forest or Neural Network) to predict redshift directly from colors, feeding estimated $z_{\text{phot}}$ into the classifier.
- **Next-Generation Surveys:** Adapt pipelines to Rubin Observatory LSST (6 filters: $u, g, r, i, z, y$) and Euclid near-infrared space photometry.
- **Deep Learning Architectures:** Investigate 1D convolutional neural networks and TabNet attention mechanisms on raw SED flux profiles.

---

## References & Course Syllabus Bibliography

1. **Zaki, M. J., & Meira, W.** (2020). *Data Mining and Machine Learning: Fundamental Concepts and Algorithms* (2nd ed.). Cambridge University Press.
2. **Gallatin, K., & Albon, C.** (2023). *Machine Learning with Python Cookbook* (2nd ed.). O'Reilly Media.
3. **Schwartz, S., & David, S. B.** (2022). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press.
4. **York, D. G., et al. [SDSS Collaboration]** (2000). The Sloan Digital Sky Survey: Technical Summary. *The Astronomical Journal*, 120(3), 1579–1587.
5. **Fukugita, M., et al.** (1996). The Sloan Digital Sky Survey Photometric System. *The Astronomical Journal*, 111, 1748.
6. **Chawla, N. V., et al.** (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
7. **Breiman, L.** (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
8. **Pedregosa, F., et al.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
