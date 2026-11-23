# X-LearnNet Unified Explainability Report

## 1. Student and prediction summary

- **Student ID:** STU012586
- **Actual test label:** 0 (No course failure)
- **Logistic Regression probability:** 0.6796 (67.96%)
- **Logistic Regression predicted class:** 1 (Failure)
- **Logistic Regression result:** False Positive
- **GraphSAGE probability:** 0.8892 (88.92%)
- **GraphSAGE predicted class:** 1 (Failure)
- **GraphSAGE result:** False Positive

Both models produce false-positive predictions for this test record. Their probabilities represent model estimates, not certainty.

## 2. Academic profile

- GPA: 2.96
- Prior GPA: 2.9
- Average course grade: 71.0
- Recorded attendance rate: 0.4500
- Calculated attendance rate: 0.7917
- Assignment submission rate: 0.33
- LMS logins in past month: 20
- Video completion rate: 0.38
- Classes attended: 38 / 48

### Data-quality warning

The recorded attendance rate and the rate calculated from attendance records
differ by 34.17 percentage points.
This discrepancy requires investigation. Neither value should silently replace the other.

## 3. Logistic Regression — SHAP

### Global feature importance

Mean absolute SHAP values across the 50 explained test samples.
These values indicate global contribution magnitude, not direction.

| Feature | Mean absolute SHAP |
|---|---:|
| attendance_rate | 0.037080 |
| assignment_submission_rate | 0.029181 |
| lms_logins_past_month | 0.024563 |
| video_completion_rate | 0.017365 |
| GPA | 0.013341 |
| prior_gpa | 0.003737 |
| average_weekly_study_hours | 0.002785 |
| avg_course_grade | 0.001778 |
| major_Undeclared | 0.001726 |
| age | 0.001605 |

### Individual SHAP contributions for STU012586

Positive contributions push the model output toward the Failure class; negative
contributions push it away from that class.

**Largest positive contributions**

| Feature | SHAP value |
|---|---:|
| attendance_rate | 0.209734 |
| assignment_submission_rate | 0.152358 |
| video_completion_rate | 0.085947 |
| GPA | 0.018035 |
| avg_course_grade | 0.013924 |
| age | 0.010213 |
| average_weekly_study_hours | 0.008608 |
| forum_participation_count | 0.005972 |
| prior_gpa | 0.004084 |
| semester_2026_Spring | 0.003884 |

**Largest negative contributions**

| Feature | SHAP value |
|---|---:|


## 4. Logistic Regression — LIME

Local surrogate-model weights for the Failure class.
Positive weights support the Failure prediction; negative weights oppose it.
These weights are local approximations, not causal effects.

**Largest positive weights**

| Feature condition | LIME weight |
|---|---:|
| attendance_rate <= -0.53 | 0.093067 |
| assignment_submission_rate <= -0.61 | 0.072809 |
| GPA <= -0.65 | 0.053070 |
| video_completion_rate <= -0.70 | 0.047535 |
| average_weekly_study_hours <= -0.90 | 0.017628 |
| -0.85 < lms_logins_past_month <= -0.09 | 0.011554 |
| cohort_year > 0.45 | 0.009952 |
| semester_2025_Fall <= -0.50 | 0.009713 |
| age <= -0.69 | 0.009710 |
| avg_course_grade <= -0.58 | 0.008555 |

**Largest negative weights**

| Feature condition | LIME weight |
|---|---:|
| major_Undeclared <= -0.06 | -0.079922 |
| gender_Not_reported <= -0.06 | -0.074966 |
| attendance_percentage <= -0.52 | -0.006012 |

## 5. GraphSAGE — GNNExplainer

### Local feature mask scores

The following are the strongest non-demographic feature mask scores for this
student. They measure the importance of features to the explained prediction;
they are not probabilities, signed effects, or causal relationships.

| Feature | Mask importance |
|---|---:|
| forum_participation_count | 0.910953 |
| prior_gpa | 0.860259 |
| avg_course_grade | 0.849321 |
| program_level_encoded_Postgraduate | 0.844677 |
| semester_encoded_2025-Fall | 0.832983 |
| assignment_submission_rate | 0.830806 |
| major_encoded_Computer Science | 0.821551 |
| major_encoded_Business | 0.814071 |
| avg_session_duration_minutes | 0.813587 |
| assessment_count | 0.813553 |

Demographic feature masks were also present in the explanation. They are
excluded from student-facing support recommendations and should be examined
in a separate fairness audit.

### Important graph edges

| Source node | Connected node | Edge mask score |
|---|---|---:|
| STU012586 | ASM-ENR0056564-FinalExam | 0.864993 |
| STU012586 | ASM-ENR0056562-FinalExam | 0.864349 |
| STU012586 | ASM-ENR0056563-Midterm | 0.863385 |
| STU012586 | ASM-ENR0056562-Assignment | 0.863355 |
| STU012586 | ASM-ENR0056564-Assignment | 0.862864 |
| STU012586 | ASM-ENR0056563-FinalExam | 0.862850 |
| STU012586 | ASM-ENR0056563-Assignment | 0.862576 |
| STU012586 | ASM-ENR0056563-Quiz | 0.862461 |
| STU012586 | ASM-ENR0056564-Midterm | 0.862420 |
| STU012586 | ASM-ENR0056565-Assignment | 0.862349 |

Edge scores are mask importance values, not probabilities or signed effects.

## 6. Integrated interpretation

Both models predicted course failure for STU012586, while the actual test label
is no course failure. Therefore, this case is a false positive for both models.

SHAP and LIME provide complementary explanations of the Logistic Regression
output. GNNExplainer provides feature and graph-edge mask information for
GraphSAGE. These methods explain model behavior; they do not establish that
the highlighted features caused an academic outcome.

The attendance discrepancy must be resolved before drawing conclusions based
on attendance. No unsupported benchmark is used to describe the student's
attendance, assignment submission, or grades as below expectations.

## 7. Suggested follow-up

- Verify the attendance records and reconcile the two attendance values.
- Review assignment and LMS records with the appropriate academic team.
- Treat the model outputs as prompts for human review, not as definitive
  judgments about the student.
- Evaluate false-positive rates and fairness across relevant student groups.
- Validate the framework on real, appropriately governed institutional data
  before operational use.

## 8. Model performance comparison

The following metrics were measured on the same held-out test set.

| Metric | Logistic Regression | GraphSAGE |
|---|---:|---:|
| Accuracy | 66.29% | 58.57% |
| Precision | 29.80% | 27.39% |
| Recall | 58.29% | 72.68% |
| F1-score | 39.43% | 39.78% |
| ROC-AUC | 0.6939 | 0.6827 |
| PR-AUC | 0.3340 | 0.3234 |

GraphSAGE achieved higher recall, identifying a greater proportion of
students who experienced course failure. Logistic Regression achieved
higher accuracy, precision, ROC-AUC and PR-AUC. Their F1-scores were
similar. Neither model is superior across every metric.

## 9. LLM-generated student explanation

The following narrative was generated using the Llama 3.2 model through
Ollama, using the supplied student profile, model predictions and GNN
explanation outputs.

> **Review note:** This is an AI-generated narrative. Numerical results
> and evidence-based sections elsewhere in this report remain authoritative.
> Recommendations require human review, particularly where data conflicts
> or unsupported academic benchmarks may exist.

**Prediction Summary**

The student, STU012586, was predicted to fail the course with a probability of 0.8892 by the GraphSAGE model and 0.6796 by the Logistic Regression model. However, the actual test label is 0, indicating no course failure.

**Academic Evidence**

The student's academic profile shows:

* GPA: 2.96 (higher than prior GPA: 2.9)
* Attendance rate: 0.45 (calculated attendance rate: 0.7916666666666666, differing by 34.17 percentage points)
* Assignment submission rate: 0.33
* Average course grade: 71.0
* LMS logins past month: 20
* Video completion rate: 0.38
* Total classes: 48
* Classes attended: 38

**Graph Explanation**

The GNNExplainer feature masks show that the following features are important for the prediction:

* forum_participation_count: 0.91095316
* prior_gpa: 0.8602589
* avg_course_grade: 0.8493212
* program_level_encoded_Postgraduate: 0.844677
* semester_encoded_2025-Fall: 0.8329827
* assignment_submission_rate: 0.83080596
* major_encoded_Computer Science: 0.82155144
* major_encoded_Business: 0.81407136
* avg_session_duration_minutes: 0.8135873
* assessment_count: 0.8135526

The GNNExplainer top edges show that the following assessment connections are important for the prediction:

* Source node: STU012586, Target node: ASM-ENR0056564-FinalExam, Importance: 0.8649933
* Source node: STU012586, Target node: ASM-ENR0056562-FinalExam, Importance: 0.86434937
* Source node: STU012586, Target node: ASM-ENR0056563-Midterm, Importance: 0.86338526
* Source node: STU012586, Target node: ASM-ENR0056562-Assignment, Importance: 0.86335516
* Source node: STU012586, Target node: ASM-ENR0056564-Assignment, Importance: 0.8628636
* Source node: STU012586, Target node: ASM-ENR0056563-FinalExam, Importance: 0.8628495
* Source node: STU012586, Target node: ASM-ENR0056563-Assignment, Importance: 0.86257565
* Source node: STU012586, Target node: ASM-ENR0056563-Quiz, Importance: 0.86246055
* Source node: STU012586, Target node: ASM-ENR0056564-Midterm, Importance: 0.8624202
* Source node: STU012586, Target node: ASM-ENR0056565-Assignment, Importance: 0.86234903

**Data Quality**

The attendance values do not match, with a discrepancy of 34.17 percentage points between the recorded attendance and the calculated attendance rate.

**Recommended Support**

Based on the data, it is recommended to investigate the attendance discrepancy and resolve it before making any further decisions. Additionally, the student's academic profile suggests that they may benefit from additional academic support to improve their course grade.

## 10. Limitations

- The current dataset is synthetic.
- The predictions are not proof of future student outcomes.
- SHAP and LIME explain the Logistic Regression model; GNNExplainer explains
  GraphSAGE. Their scores are not directly comparable.
- GNNExplainer mask scores do not provide signed feature effects.
- The current graph setup uses transductive message passing.
- The train/validation/test split has not been established as a temporal split.
- The attendance data contains a discrepancy for this student.

---
Generated by the X-LearnNet unified explainability pipeline.
