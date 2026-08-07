# Notebook libraries

- archived_at: 2026-06-11T13:49:38Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703335

Topic #703335: Notebook libraries
  Author: Alchemist
  Posted: 2026-05-29 23:49:20.073000
  Votes: 1  Comments: 1

Hi,

how can we know the list of libraries and their versions installed in the env in which the submission will run ? 
I've found this link: https://github.com/Kaggle/docker-python/blob/main/README.md

but there is no pandas or polars for example, although I see them when I launch a notebook.

Comments:
├─ PatrickAIForFun (2026-05-30 07:35:00.147000) [+1]
│  The environment during submission is the same as in the kaggle notebook session. Thus, if it works there, it will also work in the submission. If wamt a list of these, just run !pip freeze in a not...
