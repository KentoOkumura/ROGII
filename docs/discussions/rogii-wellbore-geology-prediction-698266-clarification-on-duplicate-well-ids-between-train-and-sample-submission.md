# Clarification on duplicate well IDs between train and sample_submission

- archived_at: 2026-06-11T13:50:40Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698266

Topic #698266: Clarification on duplicate well IDs between train and sample_submission
  Author: Kristof Anderson
  Posted: 2026-05-09 03:42:44.614000
  Votes: 1  Comments: 1

Good night, organizers,
While inspecting the competition data, I noticed that some well IDs appearing in sample_submission also appear in the provided train files, and the corresponding train files appear to contain TVT values for rows that are targets in the submission.
Are participants allowed to use TVT values from provided train files directly when the same well/row IDs appear in the submission set, or should those overlapping target values be treated as unavailable/leakage?
I want to make sure my approach follows the intended rules and spirit of the competition.
Thanks.

Comments:
├─ Chris Deotte (2026-05-09 08:21:52.967000) [+5]
│  Hi. Read discussion here. What we see isn't the real sample submission nor real test data. When we submit our code, the sample submission and test data gets replaced and our code sees the real ones.
