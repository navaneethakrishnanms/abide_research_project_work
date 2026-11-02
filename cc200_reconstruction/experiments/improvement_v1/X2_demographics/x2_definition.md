# X2 definition (written before running; implements improvement_plan_v1.md section 5, arm D2, without changing it)

Comparison: baseline Arm B (saved predictions; append age + sex, standardized, L2 logistic regression, fixed 0.5) versus **D2: residualization of every connectivity feature on age and sex, fitted on the training partition only, no demographics appended.**
Not run here: arm D1 (no demographics), X3, X4, leave-one-site-out. The frozen acceptance rule (plan section 4, criteria 1-4 and replication on seeds 3-5 if met) is applied unchanged; the X1-only leave-one-site-out co-requirement does not apply to X2.

Details the plan left implicit (fixed now; none changes the frozen design):
- Model per training partition: `feature_j = a_j + b_j * age_z + c_j * female + error_j`, ordinary least squares with intercept, all 19,900 features jointly; `age_z = (age - training mean) / training SD`; `female = 1 if SEX == 2 else 0`.
- Transform for any row (training or test): `feature_j - (a_j + b_j * age_z + c_j * female)`, using only that row's own age and sex and the fitted coefficients. No diagnosis label and no statistic of the rows being transformed is used.
- Missing age/sex (none in this cohort) would take the training median; the run asserts there are none.
- After residualization: StandardScaler on the 19,900 residual features, then balanced L2 logistic regression with C in {0.001, 0.01, 0.1, 1, 10} chosen by inner 5-fold ROC-AUC (as the baseline). The residualizer is refitted inside every inner fold (Pipeline), so nothing leaks inside the inner selection.
- Folds: StratifiedKFold(10, shuffle, random_state = seed) for seeds 0, 1, 2 (identical to the baseline); inner StratifiedKFold(5, shuffle, random_state = seed); fixed threshold 0.5.

Validity: the procedure is label-free and deployable (a new subject needs only age and sex); the coefficients are estimated on training rows only; held-out rows cannot influence the fit or each other (tested on synthetic data in `run_x2.py`).
Known risks (from the plan): sex differs by diagnosis, so removing sex-related feature variance may remove real signal; age is site-structured, so residualization also removes some site information.
