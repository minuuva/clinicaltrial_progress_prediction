# DS 4320 Project 2: Clinical Trial Progress Prediction — Problem Solution Pipeline

**Topic:** Predicting whether a registered clinical trial will reach completion or end early (TERMINATED / WITHDRAWN) using structured metadata available at the time of registration.

**Author:** Minu Choi | **NetID:** qce2dp

---

This notebook executes the full end-to-end pipeline:

1. **Data Preparation** — Query MongoDB Atlas and flatten nested documents into a modeling DataFrame.
2. **Feature Engineering** — Extract and encode registration-time features from document fields.
3. **Baseline** — Establish a majority-class lower bound.
4. **Modeling** — Train a Random Forest classifier and evaluate against baseline.
5. **Evaluation** — Report accuracy, macro F1, per-class F1, and confusion matrix.
6. **Visualization** — Produce a publication-quality two-panel figure.
7. **Export** — Save notebook as Markdown via nbconvert.

**Outputs:**
- `../logs/solution_pipeline.log`
- `../visualizations/pipeline_results.png`
- `solution_pipeline.md`


```python
# ── Imports ───────────────────────────────────────────────────────────────────
import logging
import os
import subprocess
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_DIR  = Path("../logs")
VIZ_DIR  = Path("../visualizations")
LOG_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "solution_pipeline.log"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info("Pipeline initialized. Log → %s", LOG_PATH)

# ── MongoDB credentials ───────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path("../.env"), override=False)
password = os.getenv("MONGODB_PASSWORD")
if not password:
    raise EnvironmentError(
        "MONGODB_PASSWORD not found. Add it to clinical_trial_progress_prediction/.env"
    )
MONGO_URI = (
    f"mongodb+srv://qce2dp:{password}@p2.trd2eby.mongodb.net/"
    "?retryWrites=true&w=majority&appName=p2"
)
DB_NAME  = "ds4320_project2"
COL_NAME = "clinical_trials"
SEED     = 42

logger.info("Configuration complete. DB=%s  Collection=%s", DB_NAME, COL_NAME)
```

    2026-04-20 22:50:45,533  INFO      Pipeline initialized. Log → ../logs/solution_pipeline.log
    2026-04-20 22:50:45,535  INFO      Configuration complete. DB=ds4320_project2  Collection=clinical_trials


## Data Preparation

### Rationale

The raw dataset lives in MongoDB Atlas as 2,100 nested JSON documents. Rather than loading every field, the query uses a MongoDB **projection** to retrieve only the 15 document paths that map to model features plus the target label. This reduces network transfer, avoids deserializing unused nested arrays, and makes the intent of the query self-documenting — the projection list is the feature specification. Raw data is not stored in the repository; Atlas is the single authoritative source, enforcing a clean separation between versioned code and managed data. Once retrieved, the documents are immediately flattened into a single `pd.DataFrame` with one row per trial, which is the standard input format expected by all downstream scikit-learn operations.


```python
# ── MongoDB projection: only fields needed for modeling ───────────────────────
PROJECTION = {
    "_id": 0,
    "protocolSection.statusModule.overallStatus": 1,
    "protocolSection.sponsorCollaboratorsModule.leadSponsor.class": 1,
    "protocolSection.designModule.phases": 1,
    "protocolSection.designModule.studyType": 1,
    "protocolSection.designModule.enrollmentInfo.count": 1,
    "protocolSection.oversightModule.isFdaRegulatedDrug": 1,
    "protocolSection.oversightModule.isFdaRegulatedDevice": 1,
    "protocolSection.armsInterventionsModule.interventions": 1,
    "protocolSection.armsInterventionsModule.armGroups": 1,
    "protocolSection.outcomesModule.primaryOutcomes": 1,
    "protocolSection.conditionsModule.conditions": 1,
    "protocolSection.contactsLocationsModule.locations": 1,
    "protocolSection.eligibilityModule.sex": 1,
    "protocolSection.eligibilityModule.healthyVolunteers": 1,
    "hasResults": 1,
}


def load_from_mongo(uri: str, db: str, collection: str) -> list[dict]:
    """
    Connect to MongoDB Atlas and return all documents using the feature projection.

    Args:
        uri:        MongoDB connection string.
        db:         Database name.
        collection: Collection name.

    Returns:
        List of projected document dicts.

    Raises:
        ValueError:       If the URI is empty or db_password is not set.
        ConnectionError:  If the MongoDB client cannot connect or authenticate.
    """

    try:
        client = MongoClient(uri, server_api=ServerApi("1"))
        client.admin.command("ping")
        docs = list(client[db][collection].find({}, PROJECTION))
        logger.info("Loaded %d documents from MongoDB Atlas.", len(docs))
        return docs
    except Exception as exc:
        raise ConnectionError(
            f"MongoDB connection failed: {exc}\n"
            "Ensure 'db_password' is set correctly and your IP is whitelisted in Atlas."
        ) from exc


# ── Load data from Atlas (sole data source — no local fallback) ───────────────
raw_docs = load_from_mongo(MONGO_URI, DB_NAME, COL_NAME)
logger.info("Total documents loaded: %d", len(raw_docs))
```

    2026-04-20 22:51:01,448  INFO      Loaded 2100 documents from MongoDB Atlas.
    2026-04-20 22:51:01,449  INFO      Total documents loaded: 2100



```python
# ── Flatten nested documents into a modeling DataFrame ────────────────────────
def extract_features(doc: dict) -> dict:
    """
    Flatten a single nested trial document into a flat feature dict.

    Args:
        doc: Raw document dict from MongoDB or local JSON.

    Returns:
        Dict with one key per feature (missing fields become None).
    """
    ps  = doc.get("protocolSection", {})
    sm  = ps.get("statusModule", {})
    sc  = ps.get("sponsorCollaboratorsModule", {})
    dm  = ps.get("designModule", {})
    om  = ps.get("oversightModule", {})
    aim = ps.get("armsInterventionsModule", {})
    out = ps.get("outcomesModule", {})
    cnd = ps.get("conditionsModule", {})
    loc = ps.get("contactsLocationsModule", {})
    elig = ps.get("eligibilityModule", {})

    phases = dm.get("phases", [])

    return {
        "status":            sm.get("overallStatus"),
        "sponsor_class":     sc.get("leadSponsor", {}).get("class"),
        "phase":             phases[0] if phases else "N/A",
        "study_type":        dm.get("studyType"),
        "enrollment":        dm.get("enrollmentInfo", {}).get("count"),
        "is_fda_drug":       om.get("isFdaRegulatedDrug"),
        "is_fda_device":     om.get("isFdaRegulatedDevice"),
        "num_interventions": len(aim.get("interventions", [])),
        "num_arms":          len(aim.get("armGroups", [])),
        "num_outcomes":      len(out.get("primaryOutcomes", [])),
        "num_conditions":    len(cnd.get("conditions", [])),
        "num_locations":     len(loc.get("locations", [])),
        "sex":               elig.get("sex"),
        "healthy_volunteers":elig.get("healthyVolunteers"),
        "has_results":       doc.get("hasResults", False),
    }


df_raw = pd.DataFrame([extract_features(d) for d in raw_docs])

logger.info("DataFrame shape: %s", df_raw.shape)
logger.info("Status counts:\n%s", df_raw["status"].value_counts().to_string())
logger.info("Missing values per column:\n%s", df_raw.isnull().sum().to_string())

print(f"Shape: {df_raw.shape}")
print(df_raw["status"].value_counts())
df_raw.head()
```

    2026-04-20 22:51:04,748  INFO      DataFrame shape: (2100, 15)
    2026-04-20 22:51:04,756  INFO      Status counts:
    status
    TERMINATED    700
    COMPLETED     700
    WITHDRAWN     700


    2026-04-20 22:51:04,758  INFO      Missing values per column:
    status                   0
    sponsor_class            0
    phase                    0
    study_type               0
    enrollment              17
    is_fda_drug           1093
    is_fda_device         1093
    num_interventions        0
    num_arms                 0
    num_outcomes             0
    num_conditions           0
    num_locations            0
    sex                      1
    healthy_volunteers      36
    has_results              0


    Shape: (2100, 15)
    status
    TERMINATED    700
    COMPLETED     700
    WITHDRAWN     700
    Name: count, dtype: int64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>status</th>
      <th>sponsor_class</th>
      <th>phase</th>
      <th>study_type</th>
      <th>enrollment</th>
      <th>is_fda_drug</th>
      <th>is_fda_device</th>
      <th>num_interventions</th>
      <th>num_arms</th>
      <th>num_outcomes</th>
      <th>num_conditions</th>
      <th>num_locations</th>
      <th>sex</th>
      <th>healthy_volunteers</th>
      <th>has_results</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TERMINATED</td>
      <td>OTHER</td>
      <td>EARLY_PHASE1</td>
      <td>INTERVENTIONAL</td>
      <td>40.0</td>
      <td>None</td>
      <td>None</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>ALL</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>COMPLETED</td>
      <td>OTHER</td>
      <td>N/A</td>
      <td>OBSERVATIONAL</td>
      <td>160.0</td>
      <td>False</td>
      <td>False</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>ALL</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TERMINATED</td>
      <td>OTHER</td>
      <td>PHASE3</td>
      <td>INTERVENTIONAL</td>
      <td>201.0</td>
      <td>False</td>
      <td>False</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>ALL</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TERMINATED</td>
      <td>OTHER</td>
      <td>N/A</td>
      <td>OBSERVATIONAL</td>
      <td>6.0</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>ALL</td>
      <td>None</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>COMPLETED</td>
      <td>OTHER</td>
      <td>NA</td>
      <td>INTERVENTIONAL</td>
      <td>32.0</td>
      <td>False</td>
      <td>False</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>ALL</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ── Feature Engineering ───────────────────────────────────────────────────────
df = df_raw.copy()

# Log-transform enrollment (right-skewed; median~15, max~5M)
df["log_enrollment"] = np.log10(df["enrollment"].clip(lower=1).fillna(1))

# Normalize all known phase labels to consistent strings
PHASE_MAP = {
    "EARLY_PHASE1": "Phase_I",
    "PHASE1":       "Phase_I",
    "PHASE2":       "Phase_II",
    "PHASE3":       "Phase_III",
    "PHASE4":       "Phase_IV",
    "NA":           "Phase_NA",
    "N/A":          "Phase_NA",
}
df["phase"] = df["phase"].map(PHASE_MAP).fillna("Phase_NA")

# Fill categorical nulls
for col in ["sponsor_class", "study_type", "sex"]:
    df[col] = df[col].fillna("UNKNOWN")

# Boolean columns: None → False
for col in ["is_fda_drug", "is_fda_device", "healthy_volunteers", "has_results"]:
    df[col] = df[col].fillna(False).astype(bool)

# Count features: None → 0
for col in ["num_interventions", "num_arms", "num_outcomes", "num_conditions", "num_locations"]:
    df[col] = df[col].fillna(0).astype(int)

# Encode target label
LABEL_MAP    = {"COMPLETED": 0, "TERMINATED": 1, "WITHDRAWN": 2}
LABEL_NAMES  = ["COMPLETED", "TERMINATED", "WITHDRAWN"]
df["label"]  = df["status"].map(LABEL_MAP)

# One-hot encode categoricals
cat_cols   = ["sponsor_class", "phase", "study_type", "sex"]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

# Feature matrix: drop non-feature columns
drop_cols    = ["status", "label", "enrollment"]
feature_cols = [c for c in df_encoded.columns if c not in drop_cols]
X = df_encoded[feature_cols].astype(float)
y = df_encoded["label"]

logger.info("Feature matrix: %d rows × %d features", *X.shape)
logger.info("Feature columns: %s", list(X.columns))
print(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
for lbl, name in enumerate(LABEL_NAMES):
    print(f"  {name}: {(y == lbl).sum()}")
```

    2026-04-20 22:51:09,415  INFO      Feature matrix shape: (2100, 27)
    2026-04-20 22:51:09,417  INFO      Feature columns: ['is_fda_drug', 'is_fda_device', 'num_interventions', 'num_arms', 'num_outcomes', 'num_conditions', 'num_locations', 'healthy_volunteers', 'has_results', 'log_enrollment', 'sponsor_class_FED', 'sponsor_class_INDUSTRY', 'sponsor_class_NETWORK', 'sponsor_class_NIH', 'sponsor_class_OTHER', 'sponsor_class_OTHER_GOV', 'phase_Phase_I', 'phase_Phase_II', 'phase_Phase_III', 'phase_Phase_IV', 'phase_Phase_NA', 'study_type_INTERVENTIONAL', 'study_type_OBSERVATIONAL', 'sex_ALL', 'sex_FEMALE', 'sex_MALE', 'sex_UNKNOWN']


    Feature matrix: 2100 rows × 27 features
    Target distribution:
    label
    0    700
    1    700
    2    700
    Name: count, dtype: int64



```python
# ── Train/Test Split and Majority-Class Baseline ──────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

logger.info("Train size: %d  |  Test size: %d", len(X_train), len(X_test))

# Majority-class baseline: always predict COMPLETED (class 0)
# This reflects the pre-balancing real-world prevalence (~86% COMPLETED on registry)
y_baseline = np.zeros(len(y_test), dtype=int)
baseline_acc = accuracy_score(y_test, y_baseline)
baseline_f1  = f1_score(y_test, y_baseline, average="macro", zero_division=0)

logger.info("Baseline accuracy: %.4f", baseline_acc)
logger.info("Baseline macro F1: %.4f", baseline_f1)

print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
print(f"Majority-class baseline  →  Accuracy: {baseline_acc:.4f}  |  Macro F1: {baseline_f1:.4f}")
```

    2026-04-20 22:51:12,212  INFO      Train size: 1680  |  Test size: 420
    2026-04-20 22:51:12,215  INFO      Baseline accuracy: 0.3333
    2026-04-20 22:51:12,216  INFO      Baseline macro F1: 0.1667


    Train: 1680 samples | Test: 420 samples
    Majority-class baseline  →  Accuracy: 0.3333  |  Macro F1: 0.1667


## Analysis

### Rationale

**Random Forest** is selected as the single classifier for this problem. It naturally handles mixed feature types — booleans, counts, log-continuous, and one-hot categoricals — without requiring feature scaling. Its ensemble of decision trees captures non-linear interactions between features like phase, sponsor class, and enrollment size that a linear model cannot represent. Crucially, it produces feature importances via mean decrease in impurity, which directly supports the interpretive goal: identifying which registration-time signals predict trial outcome. Random Forest ensemble methods are covered in DS 3021/4021.

**Macro F1** is the primary evaluation metric. All three outcome classes (COMPLETED, TERMINATED, WITHDRAWN) are equally important to the problem domain — misclassifying a withdrawn trial as completed is no less costly than the reverse. Macro F1 averages the per-class F1 score equally regardless of count, making it the correct summary metric for a balanced three-class problem.

**Stratified 80/20 split** preserves the 700/700/700 class distribution in both train and test sets, ensuring the evaluation reflects balanced performance across all outcome classes. The random seed (42) is fixed for full reproducibility.

**Log-transformation of enrollment** addresses severe right skew in the raw enrollment distribution (median ≈ 15, max > 5 M). Log-scaling compresses the tail so this single feature does not dominate tree split decisions.


```python
# ── Random Forest ─────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1  = f1_score(y_test, y_pred_rf, average="macro")

logger.info("Random Forest  →  Accuracy: %.4f  |  Macro F1: %.4f", rf_acc, rf_f1)

print("=" * 60)
print("RANDOM FOREST  (500 trees, min_samples_leaf=2)")
print("=" * 60)
print(f"Accuracy : {rf_acc:.4f}")
print(f"Macro F1 : {rf_f1:.4f}")
print()
print(classification_report(y_test, y_pred_rf, target_names=LABEL_NAMES))
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[8], line 2
          1 # ── Logistic Regression ───────────────────────────────────────────────────────
    ----> 2 lr = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs", multi_class="multinomial")
          3 lr.fit(X_train, y_train)
          4 y_pred_lr = lr.predict(X_test)
          5 


    TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'



```python
# ── Results Summary: Baseline vs Random Forest ────────────────────────────────
lift_acc = rf_acc - baseline_acc
lift_f1  = rf_f1  - baseline_f1

print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Model':<30} {'Accuracy':>10} {'Macro F1':>10}")
print("-" * 60)
print(f"{'Majority-class baseline':<30} {baseline_acc:>10.4f} {baseline_f1:>10.4f}")
print(f"{'Random Forest (500 trees)':<30} {rf_acc:>10.4f} {rf_f1:>10.4f}")
print("-" * 60)
print(f"{'Lift over baseline':<30} {lift_acc:>+10.4f} {lift_f1:>+10.4f}")

logger.info("Lift over baseline  →  Accuracy: %+.4f  |  Macro F1: %+.4f", lift_acc, lift_f1)
```


```python
# ── Per-class F1 breakdown ─────────────────────────────────────────────────────
from sklearn.metrics import f1_score as f1
per_class_f1 = f1(y_test, y_pred_rf, average=None, labels=[0, 1, 2])

print("PER-CLASS F1 SCORES")
print("-" * 35)
for name, score in zip(LABEL_NAMES, per_class_f1):
    bar = "█" * int(score * 20)
    print(f"  {name:<12} {score:.4f}  {bar}")

logger.info("Per-class F1: %s", dict(zip(LABEL_NAMES, per_class_f1.round(4))))
```

## Visualization

### Rationale

The publication-quality figure uses a two-panel design, each panel answering a distinct question a reader would have after seeing the model metrics.

**Left panel — Normalized Confusion Matrix (Random Forest):** The confusion matrix is the most informative single diagnostic for a multi-class classifier. Row-normalizing (dividing each row by its true class total) converts raw counts into recall-per-class rates, which makes per-class error patterns directly comparable regardless of class size. Color intensity encodes the rate, making misclassification concentrations immediately visible without requiring the reader to parse numbers first.

**Right panel — Top Feature Importances (Random Forest):** Feature importance from a Random Forest directly answers the scientific question motivating the project: *which registration-time signals predict trial outcome?* A horizontal bar chart is chosen over a vertical one because feature names are long strings that read naturally left-to-right. Features are colored by category (enrollment, design, sponsor, eligibility, results) so the reader can immediately see which domain of metadata contributes most, without reading every label. This chart is the interpretive payoff of the pipeline — it supports the press release claim that failure leaves a measurable fingerprint at registration.


```python
# ── Publication-Quality Two-Panel Figure ──────────────────────────────────────
TOP_N = 12

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

# ── Left: Normalized Confusion Matrix ─────────────────────────────────────────
cm      = confusion_matrix(y_test, y_pred_rf)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="YlOrBr",
    xticklabels=LABEL_NAMES,
    yticklabels=LABEL_NAMES,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "Recall Rate"},
    ax=ax1,
    vmin=0, vmax=1,
)
ax1.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
ax1.set_ylabel("True Label",      fontsize=11, labelpad=8)
ax1.set_title("Confusion Matrix\n(Random Forest, row-normalized)",
              fontsize=12, fontweight="bold", pad=10)
ax1.tick_params(axis="x", rotation=15)
ax1.tick_params(axis="y", rotation=0)

# ── Right: Feature Importance ──────────────────────────────────────────────────
importances  = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.nlargest(TOP_N).sort_values()

CATEGORY_COLORS = {
    "has_results":    "#6C757D",
    "log_enrollment": "#2A9D8F",
    "num_":           "#457B9D",
    "sponsor_class_": "#E76F51",
    "phase_":         "#264653",
    "study_type_":    "#F4A261",
    "sex_":           "#A8DADC",
    "is_fda":         "#E9C46A",
    "healthy":        "#CED4DA",
}

def get_color(name: str) -> str:
    for prefix, color in CATEGORY_COLORS.items():
        if name.startswith(prefix):
            return color
    return "#CCCCCC"

def clean_label(name: str) -> str:
    return (
        name.replace("sponsor_class_", "Sponsor: ")
            .replace("phase_", "Phase: ")
            .replace("study_type_", "Study Type: ")
            .replace("sex_", "Sex: ")
            .replace("log_enrollment", "Log Enrollment")
            .replace("num_interventions", "# Interventions")
            .replace("num_arms", "# Arms")
            .replace("num_outcomes", "# Primary Outcomes")
            .replace("num_conditions", "# Conditions")
            .replace("num_locations", "# Locations")
            .replace("is_fda_drug", "FDA-Regulated Drug")
            .replace("is_fda_device", "FDA-Regulated Device")
            .replace("healthy_volunteers", "Healthy Volunteers")
            .replace("has_results", "Has Results")
    )

bar_colors = [get_color(f) for f in top_features.index]

ax2.barh(
    range(len(top_features)),
    top_features.values,
    color=bar_colors,
    edgecolor="white",
    height=0.7,
)
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels([clean_label(f) for f in top_features.index], fontsize=9.5)
ax2.set_xlabel("Mean Decrease in Impurity", fontsize=10)
ax2.set_title(f"Top {TOP_N} Feature Importances\n(Random Forest)",
              fontsize=12, fontweight="bold", pad=10)
ax2.grid(axis="x", color="#EEEEEE", linewidth=0.8)
ax2.set_axisbelow(True)

legend_items = [
    mpatches.Patch(color="#6C757D", label="Results"),
    mpatches.Patch(color="#2A9D8F", label="Enrollment"),
    mpatches.Patch(color="#457B9D", label="Count Features"),
    mpatches.Patch(color="#E76F51", label="Sponsor"),
    mpatches.Patch(color="#264653", label="Phase"),
    mpatches.Patch(color="#F4A261", label="Study Type"),
]
ax2.legend(handles=legend_items, loc="lower right",
           fontsize=8, framealpha=0.9, title="Feature Category", title_fontsize=8.5)

fig.text(
    0.5, -0.03,
    f"Random Forest (500 trees) | Test set n={len(y_test)} | "
    f"Accuracy={rf_acc:.3f} | Macro F1={rf_f1:.3f} | "
    "Source: ClinicalTrials.gov API v2 (n=2,100)",
    ha="center", fontsize=8.5, color="#555555", style="italic",
)

plt.suptitle(
    "Predicting Clinical Trial Outcomes from Registration-Time Metadata",
    fontsize=13.5, fontweight="bold", y=1.02,
)
plt.tight_layout()

OUT_PATH = VIZ_DIR / "pipeline_results.png"
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
plt.show()

logger.info("Visualization saved → %s", OUT_PATH)
print(f"Figure saved to {OUT_PATH}")
```


```python
# ── Export notebook as Markdown ───────────────────────────────────────────────
def export_to_markdown(notebook_path: str) -> None:
    """
    Convert this notebook to a Markdown file using nbconvert.

    Args:
        notebook_path: Filename of the notebook to convert.
    """
    try:
        result = subprocess.run(
            ["jupyter", "nbconvert", "--to", "markdown", notebook_path],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info("Markdown export successful → solution_pipeline.md")
            print("Exported: solution_pipeline.md")
        else:
            logger.error("nbconvert failed: %s", result.stderr)
            print(f"nbconvert error: {result.stderr}")
    except FileNotFoundError:
        logger.warning("jupyter not found on PATH — skipping markdown export.")
        print("Warning: jupyter not found. Run manually: jupyter nbconvert --to markdown solution_pipeline.ipynb")


export_to_markdown("solution_pipeline.ipynb")
logger.info("Pipeline complete.")
print("\nPipeline complete. All outputs written.")
```
