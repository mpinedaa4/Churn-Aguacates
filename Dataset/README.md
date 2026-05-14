# Dataset

## Source and Context

This dataset was collected through a mental health survey administered to university students. The reference year for the study is **2022**, which is used to compute participant age from the recorded year of birth.

The dataset contains **2,992 observations** and **42 raw variables**. It covers demographic characteristics, subjective wellbeing ratings, and individual items from three validated psychological questionnaires: the PSS-10, GAD-7, and PHQ-9.

---

## Missing Values

Two variables contain missing values in the raw file:

| Variable | Missing Count | Handling |
|---|---|---|
| `education_years_father` | 954 | Imputed with column median during EDA |
| `education_years_mother` | 1059 | Imputed with column median during EDA |

All other variables are complete.

---

## Variable Reference

### Demographic Variables

| Variable | Type | Range / Values | Description |
|---|---|---|---|
| `year_of_birth` | Integer | 1973 – 2022 | Year the participant was born. Converted to `age` during feature engineering and then dropped. |
| `gender` | Integer (encoded) | 1, 2 | 1 = Male; 2 = Female |
| `socioeconomic_status` | Integer | 1 – 6 | Self-reported socioeconomic stratum on a 6-point national scale (1 = lowest, 6 = highest) |
| `ethnicity` | Integer (encoded) | 1 – 6 | 1 = Indigenous; 2 = Gypsy; 3 = Raizal from the San Andres, Providencia and Santa Catalina Archipelago; 4 = Palenquero from San Basilio; 5 = Black, Mulatto (Afro-descendant) or Afro-Colombian; 6 = None of the above |
| `education_years_father` | Float | 0 – 43 | Number of years of formal education completed by the participant's father |
| `education_years_mother` | Float | 0 – 43 | Number of years of formal education completed by the participant's mother |

---

### Subjective Wellbeing Variables ("Yesterday" items)

These items ask participants to report how they felt **the day before completing the survey**. Scores use two different scales depending on the question.

**Life satisfaction (single item)**

| Variable | Scale | Description |
|---|---|---|
| `life_satisfaction` | 0 – 10 | "In general, how satisfied are you with all aspects of your life?" 0 = Not satisfied at all; 10 = Totally satisfied |

**Affect items**

For all items below: **0 = At no time; 10 = All the time**

| Variable | Description |
|---|---|
| `happy_yesterday` | Felt happy |
| `laughed_yesterday` | Laughed or smiled |
| `learned_yesterday` | Learned something interesting |
| `enjoyment_yesterday` | Experienced enjoyment |
| `worried_yesterday` | Felt worried |
| `felt_depressed_yesterday` | Felt depressed |
| `angry_yesterday` | Felt angry |
| `stress_yesterday` | Felt stressed |
| `lonely_yesterday` | Felt lonely |

During feature engineering, the four positive affect items (`happy_yesterday`, `laughed_yesterday`, `learned_yesterday`, `enjoyment_yesterday`) are summed into `positive_affect`, and the five negative affect items are summed into `negative_affect`. The individual items are then dropped.

---

### PSS-10 — Perceived Stress Scale (10 items)

The PSS-10 measures the degree to which situations in one's life are appraised as stressful over the past month.

**Response scale: 0 = Never; 1 = Almost never; 2 = Sometimes; 3 = Fairly often; 4 = Very often**

**Possible total score range: 0 – 40**

| Variable | Item description |
|---|---|
| `pss_1` | Upset because of something that happened unexpectedly |
| `pss_2` | Unable to control important things in life |
| `pss_3` | Felt nervous and stressed |
| `pss_4` | Unable to deal with personal problems |
| `pss_5` | Things were going their way (reverse-scored in the original scale) |
| `pss_6` | Could not cope with everything that had to be done |
| `pss_7` | Able to control irritations (reverse-scored in the original scale) |
| `pss_8` | On top of things (reverse-scored in the original scale) |
| `pss_9` | Angered by things outside their control |
| `pss_10` | Difficulties piling up so high that could not be overcome |

Individual items are summed into `pss_total` during feature engineering and then dropped.

---

### GAD-7 — Generalised Anxiety Disorder Scale (7 items)

The GAD-7 screens for and measures severity of generalised anxiety disorder over the past two weeks.

**Response scale: 0 = Not at all; 1 = Several days; 2 = More than half the days; 3 = Nearly every day**

**Possible total score range: 0 – 21**

| Variable | Item description |
|---|---|
| `gad_1` | Feeling nervous, anxious, or on edge |
| `gad_2` | Not being able to stop or control worrying |
| `gad_3` | Worrying too much about different things |
| `gad_4` | Trouble relaxing |
| `gad_5` | Being so restless it is hard to sit still |
| `gad_6` | Becoming easily annoyed or irritable |
| `gad_7` | Feeling afraid as if something awful might happen |

Individual items are summed into `gad_total` during feature engineering and then dropped.

---

### PHQ-9 — Patient Health Questionnaire (9 items)

The PHQ-9 screens for and measures the severity of depression over the past two weeks. It is the primary outcome variable of this project.

**Response scale: 0 = Not at all; 1 = Several days; 2 = More than half the days; 3 = Nearly every day**

**Possible total score range: 0 – 27**

| Variable | Item description |
|---|---|
| `phq_1` | Little interest or pleasure in doing things |
| `phq_2` | Feeling down, depressed, or hopeless |
| `phq_3` | Trouble falling or staying asleep, or sleeping too much |
| `phq_4` | Feeling tired or having little energy |
| `phq_5` | Poor appetite or overeating |
| `phq_6` | Feeling bad about yourself, or that you are a failure |
| `phq_7` | Trouble concentrating on things |
| `phq_8` | Moving or speaking so slowly that others could have noticed, or being fidgety |
| `phq_9` | Thoughts that you would be better off dead or of hurting yourself |

Individual items are summed into `phq9_total` and then categorised into `phq9_category` using the standard clinical thresholds below. Both the items and the numeric total are used in the pipeline; the categorical label serves as the primary classification target.

---

## Target Variable: PHQ-9 Severity Categories

The `phq9_category` variable is derived from `phq9_total` using the standard Kroenke et al. (2001) thresholds:

| Category | Score Range | Clinical Interpretation |
|---|---|---|
| Minimal | 0 – 4 | Minimal or no depressive symptoms |
| Mild | 5 – 9 | Mild depressive symptoms |
| Moderate | 10 – 14 | Moderate depressive symptoms |
| Moderately_Severe | 15 – 19 | Moderately severe depressive symptoms |
| Severe | 20 – 27 | Severe depressive symptoms |

---

## Variables Present After Feature Engineering

Once the EDA pipeline runs, the raw 42-column dataset is transformed into the following engineered variables:

| Variable | Derived From | Type |
|---|---|---|
| `age` | `year_of_birth` | Integer |
| `gender` | — | Integer (encoded) |
| `socioeconomic_status` | — | Integer |
| `ethnicity` | — | Integer (encoded) |
| `education_years_father` | — | Float (imputed) |
| `education_years_mother` | — | Float (imputed) |
| `life_satisfaction` | — | Integer |
| `positive_affect` | Sum of 4 wellbeing items | Integer |
| `negative_affect` | Sum of 5 negative affect items | Integer |
| `pss_total` | Sum of `pss_1` to `pss_10` | Integer |
| `gad_total` | Sum of `gad_1` to `gad_7` | Integer |
| `phq9_total` | Sum of `phq_1` to `phq_9` | Integer |
| `phq9_category` | Binned from `phq9_total` | String (categorical) |