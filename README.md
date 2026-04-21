# DS 4320 Project 2: Clinical Trial Progress Prediction

## Executive Summary

This repository contains a three-class classification pipeline that predicts whether a registered clinical trial will reach completion or end early (TERMINATED or WITHDRAWN) using structured metadata available at the time of trial registration. The dataset consists of 2,100 balanced JSON documents — 700 per outcome class — stored in MongoDB Atlas and acquired via the ClinicalTrials.gov API v2. A Random Forest classifier (500 trees) achieves 78.6% accuracy and macro F1 of 0.784 on a held-out test set, a lift of +0.45 over a majority-class baseline. WITHDRAWN trials are classified with near-perfect F1 (0.97), while COMPLETED and TERMINATED trials exhibit mutual confusion (F1 of 0.71 and 0.67 respectively), consistent with their shared interventional trial structure. The repository includes data acquisition code, a full analysis and visualization pipeline, and a press release.

**Author:** Minu Choi  
**NetID:** qce2dp  
**DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19674339.svg)](https://doi.org/10.5281/zenodo.19674339)
**License:** [MIT License](LICENSE)

### Quick Links
- **Press Release:** [Press Release](PRESS_RELEASE.md)
- **Pipeline:** [Analysis Pipeline](code/solution_pipeline.ipynb)

---

## Problem Definition

### General Problem: 

Clinical drug trials — General Problem #8 from the DS 4320 Project 2 rubric.

### Refined Specific Problem:

Predicting whether a registered clinical trial will reach **completion** or end
early (**terminated** or **withdrawn**) using structured metadata available at
the time of trial registration — framed as a three-class classification problem
(COMPLETED / TERMINATED / WITHDRAWN) evaluated against a majority-class baseline,
with model performance reported via accuracy, macro F1, and a confusion matrix.

### Rationale for Refinement

The general problem of "clinical drug trials" encompasses a broad landscape of
analytical questions: safety signal detection, adverse event prediction, optimal
dosing, patient stratification, and trial operations management, among others.
I narrowed the focus to **trial completion prediction** for three specific
reasons. First, it is the problem where structured registration metadata — the
data ClinicalTrials.gov makes freely available through its public API — is most
directly informative; fields like enrollment size, sponsor class, study phase,
and intervention type are documented before the trial begins and are plausibly
causal drivers of whether a trial finishes. Second, completion status is a
clean, well-defined outcome label: ClinicalTrials.gov enforces a controlled
vocabulary (COMPLETED, TERMINATED, WITHDRAWN) that requires no subjective
annotation, making the target variable reliable across all 2,100 documents in
our dataset. Third, this framing maps naturally onto the document model — each
trial is a self-contained JSON document with deeply nested arrays for sponsors,
conditions, interventions, outcomes, and locations, producing genuine schema
variation across documents that demonstrates the document model's advantages
over a relational schema. Together these three factors — feature availability
at prediction time, label reliability, and natural document structure — make
trial completion prediction the most feasible and scientifically coherent
refinement of the general problem.

### Motivation
Clinical trials are the cornerstone of evidence-based medicine, yet an estimated
50% of registered trials never reach their primary completion date. A single
Phase III trial can cost upward of $300 million and take a decade to complete;
early termination wastes those resources, delays access to potentially life-saving
treatments, and exposes enrolled patients to experimental interventions with no
resulting benefit to science. The problem is not rare — ClinicalTrials.gov, the
world's largest public registry, lists over 33,000 terminated and 16,000 withdrawn
studies as of early 2026. Despite this scale, trial sponsors, regulatory bodies,
and funding agencies currently have no systematic, data-driven tool to assess
completion risk at registration. A predictive model trained on the structural
metadata that every trial is required to report — sponsor type, phase, enrollment
size, intervention category, eligibility criteria, and study design — could flag
high-risk trials before resources are committed, enabling sponsors to redesign
protocols, strengthen recruitment plans, or reallocate funding toward studies
with a higher probability of generating usable evidence.

**Press Release**

**Headline:** New Model Predicts Clinical Trial Failure Before It Happens — Using Only Data Available on Day One

**Full Press Release:** [PRESS_RELEASE.md](PRESS_RELEASE.md)

---

## Domain Exposition

### Terminology
| **Term** | **Definition** |
|----------|----------------|
| **Clinical Trial** | A research study that tests how well a medical intervention (drug, device, or procedure) works in human participants under controlled conditions |
| **ClinicalTrials.gov** | The U.S. National Library of Medicine's public registry of clinical studies, mandated by law for most federally funded and FDA-regulated trials |
| **NCT ID** | ClinicalTrials.gov identifier (e.g., NCT00123456) — the unique primary key for every registered study |
| **Overall Status** | The current lifecycle state of a trial as reported to ClinicalTrials.gov; controlled vocabulary includes COMPLETED, TERMINATED, WITHDRAWN, RECRUITING, etc. |
| **COMPLETED** | Trial finished with all primary outcome data collected as originally planned |
| **TERMINATED** | Trial stopped early by the sponsor or investigators after enrollment began (e.g., safety signal, futility, loss of funding) |
| **WITHDRAWN** | Trial cancelled before any participants were enrolled |
| **Phase** | FDA-defined stage of drug development: Phase I (safety/dosing), Phase II (efficacy signal), Phase III (confirmatory), Phase IV (post-market surveillance) |
| **Enrollment** | The target or actual number of participants recruited into a trial |
| **Sponsor** | The entity legally responsible for initiating and managing the trial; classified as INDUSTRY, NIH, FED, OTHER, or INDIV |
| **Lead Sponsor Class** | Categorical label for the type of organization funding and overseeing the trial |
| **Intervention Type** | The category of experimental treatment being studied: DRUG, DEVICE, BIOLOGICAL, PROCEDURE, BEHAVIORAL, etc. |
| **Study Type** | Broad classification of the study design: INTERVENTIONAL (tests a treatment) or OBSERVATIONAL (observes without intervention) |
| **Phase III Trial** | The pivotal confirmatory stage; most expensive and resource-intensive, typically requiring thousands of participants |
| **Primary Outcome** | The main endpoint(s) the trial is designed and powered to measure |
| **Eligibility Criteria** | Inclusion and exclusion rules defining who may participate in the trial |
| **Arms** | Distinct groups within a trial (e.g., treatment arm vs. placebo arm) that receive different interventions |
| **IND (Investigational New Drug)** | FDA authorization allowing an unapproved drug to be tested in humans |
| **FDA-Regulated Drug** | Boolean flag on ClinicalTrials.gov indicating whether the intervention falls under FDA drug regulation |
| **DMC / DSMB** | Data Monitoring Committee / Data Safety Monitoring Board — independent body that reviews accumulating trial data and can recommend early stopping |
| **Protocol** | The formal document defining a trial's objectives, design, methodology, statistical analysis plan, and oversight |
| **Trial Completion Rate** | Percentage of registered trials that reach COMPLETED status; a KPI for sponsor reliability and protocol quality |
| **Termination Rate** | Percentage of enrolled trials that end in TERMINATED status; a KPI for identifying high-risk sponsor types or therapeutic areas |
| **Macro F1** | Evaluation metric averaging F1 score equally across all classes regardless of class size; preferred when classes are balanced |
| **Accuracy** | Fraction of correctly classified trials out of all trials; meaningful here because the dataset is class-balanced (33.3% per class) |
| **Majority-Class Baseline** | Naïve classifier that always predicts the most frequent class; serves as the lower bound for model performance |
| **Confusion Matrix** | Table showing predicted vs. actual class counts across all three status labels; used to identify which misclassification types are most common |
| **Feature Engineering** | Process of deriving model input variables from raw nested document fields (e.g., extracting enrollment count from `designModule.enrollmentInfo`) |

### Domain Overview
This project operates within the domain of **clinical trial operations and translational medicine informatics** — the intersection of biomedical research management, regulatory science, and applied machine learning. Clinical trials are the legally mandated pathway through which new drugs, biologics, and medical devices move from laboratory discovery to patient use, governed in the United States primarily by the FDA and the 2007 FDA Amendments Act, which established the requirement for trial registration on ClinicalTrials.gov. The domain is characterized by extreme resource asymmetry: early-phase trials may cost tens of millions of dollars, while late-stage Phase III trials routinely exceed $300 million and span five to ten years, making operational failure — termination or withdrawal — one of the most expensive outcomes in all of science. The key organizational actors include pharmaceutical and biotechnology companies (INDUSTRY sponsors), federal agencies such as the NIH and Department of Veterans Affairs (NIH/FED sponsors), and academic medical centers (OTHER sponsors), each operating under different incentive structures and risk tolerances that influence how trials are designed and ultimately completed. Data science enters this domain through pharmacovigilance, patient recruitment optimization, and — as in this project — predictive modeling of trial metadata to forecast operational outcomes. ClinicalTrials.gov serves as the canonical data infrastructure, providing a standardized public registry of over 500,000 studies with structured metadata collected under consistent schema guidelines since 2000, making it one of the richest open datasets in biomedical research.

### Background Reading
All background reading materials are available in this OneDrive folder: [Background Readings Folder](https://myuva-my.sharepoint.com/:f:/g/personal/qce2dp_virginia_edu/IgBL8734dT6IR6cAiK-yILtgAbk3DUFRe7gy9oWAvAtjG_w?e=5lf3Xc).


### Summary Table of Background Readings

| **Title** | **Brief Description** | **Link** |
|-----------|----------------------|----------|
| Understanding and predicting COVID-19 clinical trial completion vs. cessation | Applies ML to 4,441 COVID-19 trials from ClinicalTrials.gov to predict completion vs. cessation; achieves 0.87 AUC using drug features and study keywords. Closest methodological parallel to this project. | [Elkin & Zhu, 2021](https://myuva-my.sharepoint.com/:b:/g/personal/qce2dp_virginia_edu/IQCe1jHtR1W1Tr4E5xQJ7yr1AfkgqxLP2JT5j7q2jGkmI7c?e=kzoS6V) |
| Predicting Phase 1 Lymphoma Clinical Trial Durations Using Machine Learning | Uses Random Forest on 1,089 Phase 1 lymphoma trials from ClinicalTrials.gov to predict trial duration adherence; achieves 0.77 AUC, demonstrating cross-disease generalizability. | [Long et al., 2023](https://myuva-my.sharepoint.com/:b:/g/personal/qce2dp_virginia_edu/IQAtzRwZLKkrTKvyD3lXI7ocAR9tEFOb7iHzYsIa2SBWZdA?e=KEpTlY) |
| Approaches in analyzing predictors of trial failure: a scoping review and meta-epidemiological study | Comprehensive 2026 scoping review of 81 studies analyzing predictors of trial failure; finds 90% used ClinicalTrials.gov data and recommends classifying TERMINATED and WITHDRAWN as failed and COMPLETED as non-failed — the same labeling scheme used in this project. | [Jovanovic et al., 2026](https://myuva-my.sharepoint.com/:b:/g/personal/qce2dp_virginia_edu/IQA_3BqoUZTyTLiVUj_RXXjMAROTR80-3sG4nyT67yvelRU?e=8idaHv) |
| Can we predict trial failure among older adult-specific clinical trials using trial-level factors? | Evaluates eight ML algorithms on phase 2–4 cancer trials using sponsor type, number of centers, arms, and enrollment size as predictors; identifies trial-level features accessible at registration that drive failure. | [Lee et al., 2023](https://myuva-my.sharepoint.com/:b:/g/personal/qce2dp_virginia_edu/IQBk8k3BbwakR4WyQWTjai3CATZbeVLTUVOsRBZwVU-YypI?e=yFSEaD) |
| Estimation of clinical trial success rates and related parameters | Analyzes 406,038 clinical trial entries across 21,143 compounds to estimate phase-by-phase success rates; foundational reference establishing that overall trial success is well below 50% and varies significantly by disease area and sponsor type. | [Wong et al., 2019](https://myuva-my.sharepoint.com/:b:/g/personal/qce2dp_virginia_edu/IQBi4wpIgrjuQJUl--R5xcOWARBGybrWhTDXySE3pCbVjvM?e=Bu8oWl) |
---

## Data Creation

### Data Provenance

The dataset was acquired on **April 7, 2026** from the **ClinicalTrials.gov API v2**
(`https://clinicaltrials.gov/api/v2/studies`), the official public REST API maintained by the U.S. National Library of Medicine. The API requires no authentication and returns structured JSON — making each response a native document ready for MongoDB ingestion. To address the natural class imbalance on the registry (COMPLETED trials represent ~86% of all resolved studies), three separate paginated requests were issued — one per status (`COMPLETED`, `TERMINATED`, `WITHDRAWN`) — each capped at 700 documents, yielding **2,100 trials balanced exactly 33.3% per class**. Documents are the complete, unmodified JSON objects returned by the API, shuffled with a fixed seed of 42 for reproducibility. All acquisition logic,
pagination, error handling, logging, and verification are implemented in
`code/data_acquisition.ipynb`.

### Code Documentation

| **File** | **Description** | **Link** |
|----------|-----------------|----------|
| `code/data_acquisition.ipynb` | Pulls 2,100 clinical trial documents from the ClinicalTrials.gov API v2 in three balanced, paginated requests (700 each for COMPLETED, TERMINATED, WITHDRAWN); loads documents directly into MongoDB Atlas; runs scale, structural, and class-balance verification checks with full logging to `logs/data_acquisition.log`. Raw data is not stored in the repository. | [data_acquisition.ipynb](code/data_acquisition.ipynb) |

### Bias Identification

Several biases are inherent to collecting from ClinicalTrials.gov. **Reporting bias** exists
because terminated and withdrawn trials are systematically under-reported. Sponsors are not
always compliant with mandatory result-reporting requirements, meaning failed trials are
underrepresented relative to their true prevalence. **Selection bias** arises from the
API's default sort order: our 700 documents per class were drawn from the first pages
returned, which skew toward older, more established studies rather than a random cross-section
of the registry. **Sponsor representation bias** exists because industry-funded trials
dominate the registry, so patterns learned by the model may reflect pharmaceutical company
behavior more than academic or NIH-funded research. Finally, **temporal bias** is present
because trial design norms, regulatory requirements, and reporting standards have shifted
substantially since 2000, meaning older and newer trials are not directly comparable.

### Bias Mitigation

The most direct mitigation applied during acquisition was **class balancing**, capping each
outcome class at exactly 700 documents so that COMPLETED, TERMINATED, and WITHDRAWN trials
each represent 33.3% of the dataset. Without this, COMPLETED trials would constitute ~86%
of the collection and any classifier would achieve high accuracy simply by predicting
COMPLETED for every trial. In the analysis pipeline, a **majority-class baseline** is
reported alongside model metrics so that any performance gain is measured against the
trivial solution. Temporal bias can be partially addressed by including study submission year
as a feature or by restricting the training window to a recent date range. Sponsor
representation bias can be quantified by disaggregating model performance metrics by sponsor
class (INDUSTRY vs. NIH vs. OTHER) to check whether the model generalizes across funding
types or overfits to the dominant industry pattern.

### Rationale for Critical Decisions

**Class Balancing at 700 per Status:** The registry contains ~316,000 COMPLETED, ~33,000
TERMINATED, and ~16,000 WITHDRAWN trials, roughly a 20:2:1 ratio. Training a classifier
on this natural distribution would produce a model that nearly always predicts COMPLETED and
still achieves high accuracy, which is scientifically useless. Capping at 700 per class
creates a balanced training problem where the model must learn genuine structural differences
between outcomes rather than exploit base-rate frequencies. The tradeoff is that the
dataset no longer reflects real-world prevalence, so predicted probabilities cannot be
interpreted as true population likelihoods without recalibration.

**700 as the Target Count:** 700 was chosen because it is well above the 1,000-document
rubric threshold when summed across three classes (2,100 total), provides enough per-class
examples for robust cross-validation, and sits comfortably within the WITHDRAWN class
ceiling (~16,000 available). This ensures the balancing does not exhaust the smallest class.
A larger target would have been feasible but offers diminishing returns for a
proof-of-concept classifier.

**No Field Filtering During Acquisition:** Documents are stored as complete, unmodified API
responses with all nested fields intact. This preserves optionality; the pipeline can
select any combination of features without re-querying the API. It also ensures the MongoDB
collection serves as a faithful replica of the source data. The tradeoff is a larger
storage footprint (~61 MB) and the need for robust null-handling in the pipeline.

**API Default Sort Order:** Documents were collected in the order returned by the API
without applying a randomization parameter at query time. This introduces the selection
bias noted above (earlier-registered studies appear first) but was chosen for
reproducibility: the same query on the same date will always return the same documents.
Adding a random sort at the API level would make exact reproduction harder without
storing the full result set.

---

## Metadata

### Implicit Schema

Each document in the `clinical_trials` collection corresponds to one registered study. Documents follow a consistent top-level
structure but exhibit genuine schema variation at lower levels. This means some modules are present
in every document, others appear only for specific study types (ex: `designInfo` fields
differ between interventional and observational studies, and `armGroups` may be absent for
single-arm trials). This variability is the natural soft-schema behavior the document model
is designed to accommodate.
**Always present (every document):**
- `protocolSection.identificationModule.nctId` — unique study identifier (primary key)
- `protocolSection.statusModule.overallStatus` — outcome label (COMPLETED / TERMINATED / WITHDRAWN)
- `protocolSection.sponsorCollaboratorsModule.leadSponsor` — sponsor name and class
**Commonly present (most documents):**
- `protocolSection.designModule.enrollmentInfo.count` — planned participant count
- `protocolSection.designModule.phases` — array of phase labels (e.g., `["PHASE2"]`)
- `protocolSection.conditionsModule.conditions` — array of condition/disease strings
- `protocolSection.armsInterventionsModule.interventions` — array of intervention objects
- `protocolSection.eligibilityModule` — sex, age range, healthy volunteer flag
- `protocolSection.outcomesModule.primaryOutcomes` — array of primary endpoint objects
- `derivedSection.conditionBrowseModule.meshes` — MeSH-standardized condition terms
**Variable / study-type dependent:**
- `protocolSection.armsInterventionsModule.armGroups` — present only for multi-arm trials
- `protocolSection.designModule.designInfo` — subfields differ between INTERVENTIONAL
  and OBSERVATIONAL study types
- `protocolSection.contactsLocationsModule.locations` — absent for many withdrawn trials
  that never established sites
- `protocolSection.descriptionModule.detailedDescription` — optional free-text field
- `derivedSection.interventionBrowseModule` — present only when intervention maps to a
  known drug or device MeSH term
- `hasResults` — boolean; `true` only for COMPLETED trials that submitted result summaries
**Example skeleton (abbreviated):**
```json
{
  "protocolSection": {
    "identificationModule": { "nctId": "NCT00000000", "briefTitle": "..." },
    "statusModule":         { "overallStatus": "COMPLETED", "startDateStruct": {"date": "2010-01"} },
    "sponsorCollaboratorsModule": {
      "leadSponsor": { "name": "...", "class": "INDUSTRY" },
      "collaborators": [ { "name": "...", "class": "NIH" } ]
    },
    "designModule": {
      "studyType": "INTERVENTIONAL",
      "phases": ["PHASE3"],
      "enrollmentInfo": { "count": 450, "type": "ACTUAL" }
    },
    "conditionsModule":        { "conditions": ["Type 2 Diabetes"] },
    "armsInterventionsModule": { "interventions": [ {"type": "DRUG", "name": "..."} ] },
    "eligibilityModule":       { "sex": "ALL", "stdAges": ["ADULT", "OLDER_ADULT"] }
  },
  "derivedSection": {
    "conditionBrowseModule": { "meshes": [ {"id": "...", "term": "Diabetes Mellitus"} ] }
  },
  "hasResults": true
}
```

### Data Tables

| **Attribute** | **Value** |
|---------------|-----------|
| Collection name | `clinical_trials` |
| Total documents | 2,100 |
| COMPLETED documents | 700 (33.3%) |
| TERMINATED documents | 700 (33.3%) |
| WITHDRAWN documents | 700 (33.3%) |
| Total serialized size | ~61.5 MB |
| Source | ClinicalTrials.gov API v2 |
| Acquisition date | April 7, 2026 |
| Document structure | Nested JSON (3–15 populated modules per document) |
| Storage | MongoDB Atlas (`clinical_trials` collection) |

### Data Dictionary

| **Field** | **JSON Path** | **Data Type** | **Description** | **Example** |
|-----------|--------------|---------------|-----------------|-------------|
| `nctId` | `protocolSection.identificationModule.nctId` | String | Unique ClinicalTrials.gov identifier; primary key | `"NCT00830635"` |
| `briefTitle` | `protocolSection.identificationModule.briefTitle` | String | Short human-readable title of the study | `"Aspirin in High-Risk Patients"` |
| `overallStatus` | `protocolSection.statusModule.overallStatus` | String (categorical) | Trial outcome — target label for classification | `"COMPLETED"` |
| `startDate` | `protocolSection.statusModule.startDateStruct.date` | String (date) | Date the trial began enrolling participants | `"2010-03"` |
| `primaryCompletionDate` | `protocolSection.statusModule.primaryCompletionDateStruct.date` | String (date) | Date primary outcome data collection was completed | `"2015-06"` |
| `studyFirstSubmitDate` | `protocolSection.statusModule.studyFirstSubmitDate` | String (date) | Date the trial was first registered on ClinicalTrials.gov | `"2009-01-27"` |
| `leadSponsorName` | `protocolSection.sponsorCollaboratorsModule.leadSponsor.name` | String | Name of the primary sponsoring organization | `"Pfizer"` |
| `leadSponsorClass` | `protocolSection.sponsorCollaboratorsModule.leadSponsor.class` | String (categorical) | Type of sponsoring organization | `"INDUSTRY"` |
| `isFdaRegulatedDrug` | `protocolSection.oversightModule.isFdaRegulatedDrug` | Boolean | Whether the intervention is an FDA-regulated drug | `true` |
| `isFdaRegulatedDevice` | `protocolSection.oversightModule.isFdaRegulatedDevice` | Boolean | Whether the intervention is an FDA-regulated device | `false` |
| `studyType` | `protocolSection.designModule.studyType` | String (categorical) | Broad study classification | `"INTERVENTIONAL"` |
| `phases` | `protocolSection.designModule.phases` | Array of strings | Trial phase(s) as defined by FDA | `["PHASE3"]` |
| `enrollmentCount` | `protocolSection.designModule.enrollmentInfo.count` | Integer | Planned or actual number of participants | `450` |
| `enrollmentType` | `protocolSection.designModule.enrollmentInfo.type` | String (categorical) | Whether enrollment count is estimated or actual | `"ESTIMATED"` |
| `conditions` | `protocolSection.conditionsModule.conditions` | Array of strings | Disease(s) or condition(s) under study | `["Type 2 Diabetes"]` |
| `interventionTypes` | `protocolSection.armsInterventionsModule.interventions[].type` | Array of strings | Category of each intervention | `["DRUG", "PLACEBO"]` |
| `interventionNames` | `protocolSection.armsInterventionsModule.interventions[].name` | Array of strings | Name of each intervention | `["Metformin", "Placebo"]` |
| `armGroups` | `protocolSection.armsInterventionsModule.armGroups` | Array of objects | Distinct treatment arms in the trial | `[{"label": "Treatment", "type": "EXPERIMENTAL"}]` |
| `primaryOutcomes` | `protocolSection.outcomesModule.primaryOutcomes` | Array of objects | Primary endpoint measure objects | `[{"measure": "HbA1c reduction", "timeFrame": "12 weeks"}]` |
| `sex` | `protocolSection.eligibilityModule.sex` | String (categorical) | Eligible participant sex | `"ALL"` |
| `stdAges` | `protocolSection.eligibilityModule.stdAges` | Array of strings | Eligible age groups | `["ADULT", "OLDER_ADULT"]` |
| `healthyVolunteers` | `protocolSection.eligibilityModule.healthyVolunteers` | Boolean | Whether healthy volunteers are accepted | `false` |
| `locations` | `protocolSection.contactsLocationsModule.locations` | Array of objects | Participating site objects with country and city | `[{"country": "United States", "city": "Boston"}]` |
| `conditionMeshTerms` | `derivedSection.conditionBrowseModule.meshes[].term` | Array of strings | NLM-standardized MeSH condition vocabulary | `["Diabetes Mellitus, Type 2"]` |
| `hasResults` | `hasResults` | Boolean | Whether the sponsor submitted result summaries | `true` |

### Data Dictionary: Uncertainty Quantification for Numerical Features

| **Feature** | **n** | **Mean** | **Std Dev** | **CV (%)** | **Min** | **Max** | **Range** | **Median** | **IQR** | **95% CI** |
|-------------|-------|----------|-------------|------------|---------|---------|-----------|------------|---------|------------|
| `enrollmentCount` | 2,088 | 2,760.14 | 114,729.17 | 4,156.6 | 0 | 5,239,194 | 5,239,194 | 15.0 | 60.0 | [-2,161.0, 7,681.3] |
| `numConditions` | 2,100 | 1.83 | 3.49 | 190.9 | 1 | 124 | 123 | 1.0 | 1.0 | [1.68, 1.97] |
| `numInterventions` | 1,944 | 1.94 | 1.42 | 73.2 | 1 | 28 | 27 | 2.0 | 1.0 | [1.88, 2.00] |
| `numPrimaryOutcomes` | 2,029 | 1.84 | 2.46 | 134.0 | 1 | 47 | 46 | 1.0 | 1.0 | [1.73, 1.94] |
| `numLocations` | 1,767 | 6.00 | 24.58 | 409.5 | 1 | 684 | 683 | 1.0 | 1.0 | [4.86, 7.15] |
| `numArmGroups` | 1,855 | 1.97 | 1.08 | 54.9 | 1 | 16 | 15 | 2.0 | 1.0 | [1.92, 2.02] |

**Statistical Interpretation:**

**Enrollment Extreme Skew (CV=4,157%):** The mean enrollment (2,760) is completely
unrepresentative — one outlier trial targeting 5.2 million participants drives the mean and
produces a 95% CI that extends into negative values, which is statistically nonsensical for
a count variable. The median (15 participants) and IQR (60) are the only reliable central
tendency measures. In the pipeline, `enrollmentCount` will be log-transformed before use
as a model feature to compress this extreme right-skew.

**Condition Count Sparsity (CV=191%):** Most trials list exactly one condition (median=1),
but a small number of broad-scope trials list up to 124, producing high variance. The narrow
95% CI [1.68, 1.97] reflects adequate sample size, not low dispersion.

**Intervention and Arm Consistency (CV=73%, 55%):** These are the most stable numerical
features. Most trials have 1–2 interventions and 1–2 arms, consistent with standard
parallel-group designs. The low CV relative to other features makes these reliable
predictors.

**Location Count Sparsity (CV=410%):** The median of 1 location (single-site trial)
versus a maximum of 684 reflects the spectrum from small investigator-initiated studies to
large multinational trials. WITHDRAWN trials are disproportionately absent from this field
(1,767 of 2,100 documents have locations), as many were cancelled before sites were
established.

**Primary Outcomes Sparsity (CV=134%):** Nearly all trials define exactly one primary
outcome (median=1), but complex adaptive trials can define up to 47. The high CV is driven
entirely by rare extreme values in the right tail.

---

## Repository Structure
```text
[todo]
```