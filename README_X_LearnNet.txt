# X-LearnNet synthetic educational dataset bundle

## Contents
- students_master.csv: one cleaned row per student, 20,000 rows.
- course_catalog.csv: course metadata.
- faculty.csv: synthetic faculty metadata.
- course_enrollments.csv: student-course-faculty-semester relationships.
- assessment_results.csv: quiz, assignment, midterm and final exam records.
- attendance_records.csv: session-level attendance.
- lms_activity_events.csv: timestamped LMS events for sequential modelling.
- weekly_learning_sequences.csv: weekly aggregates for Transformer/temporal models.
- knowledge_graph_edges.csv: source-edge-target graph edge list for GNN/GraphSAGE/Node2Vec.
- learning_resource_catalog.csv: resource metadata for recommendation/RAG integration.
- intervention_outcomes.csv: simulated intervention records and follow-up scores.
- student_outcomes_targets.csv: simulated future academic outcome labels.

## Important research integrity note
All records in this bundle are synthetic or derived from the previously supplied synthetic dataset. They are suitable for pipeline development, code testing, schema design and demonstrations. They are NOT real student records and must not be presented as real-world evidence. The future outcome labels and intervention effects are simulated and cannot establish predictive validity or causal impact. For a publishable evaluation, replace or validate these labels using appropriately consented, anonymized institutional data, and use student-level/time-based train-validation-test splits to prevent leakage.

## Suggested joins
- students_master.student_id = course_enrollments.student_id
- course_catalog.course_id = course_enrollments.course_id
- faculty.faculty_id = course_enrollments.faculty_id
- course_enrollments.enrollment_id = assessment_results.enrollment_id / attendance_records.enrollment_id
- students_master.student_id = lms_activity_events.student_id / weekly_learning_sequences.student_id / student_outcomes_targets.student_id / intervention_outcomes.student_id
- knowledge_graph_edges.csv uses node IDs from the relevant entity tables plus assessment/resource node IDs.

## Key modelling cautions
- Predict future outcomes only from features timestamped before the prediction cutoff.
- Do not use next_semester_GPA, course_failure_flag or dropout_flag as input features when predicting those same outcomes.
- Do not use risk_level as both a label and an input feature in the same risk prediction experiment.
- Use SHAP/LIME on held-out predictions; explain model behaviour, not causal reasons.
- RAG should retrieve approved syllabus, notes, policies and learning resources. The resource catalog here is metadata only; populate it with actual authorized content.
- Evaluate class imbalance using precision, recall, F1, ROC-AUC and PR-AUC; report calibration and subgroup performance where appropriate.
