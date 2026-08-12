# Kaggle Badge Catalog (55 badges)

Complete reference of all Kaggle badges with earning criteria and workflow support.

## Phase 1: Instant API (~16 badges, 5-10 min)

| Badge | Category | How to Earn | Method |
|-------|----------|-------------|--------|
| Python Coder | Notebooks | Push a Python notebook via API | `kaggle kernels push` |
| R Coder | Notebooks | Push an R notebook via API | `kaggle kernels push` (language=r) |
| API Notebook Creator | Notebooks | Create a notebook using the Kaggle API | `kaggle kernels push` |
| Utility Scripter | Notebooks | Push a utility script (kernel_type=script) | `kaggle kernels push` |
| Code Uploader | Notebooks | Upload code to Kaggle | `kaggle kernels push` |
| Code Forker | Notebooks | Fork an existing public notebook | `kaggle kernels pull` + `push` |
| Code Tagger | Notebooks | Add tags/keywords to a notebook | keywords in kernel-metadata.json |
| Dataset Creator | Datasets | Create a new dataset | `kaggle datasets create` |
| API Dataset Creator | Datasets | Create a dataset via API | `kaggle datasets create` |
| Dataset Tagger | Datasets | Add tags to a dataset | keywords in dataset-metadata.json |
| Dataset Documenter | Datasets | Achieve usability score 10/10 | Full metadata + schema + README |
| Model Creator | Models | Create a new model | `kaggle models create` |
| API Model Creator | Models | Create a model via API | `kaggle models create` |
| Model Variation Creator | Models | Create a model variation | `kaggle models variations versions create` |
| Model Tagger | Models | Add tags to a model | `kagglehub.model_upload()` |
| Model Documenter | Models | Achieve usability score 10/10 | Full metadata + description + README |

## Phase 2: Competition (~7 badges, 10-15 min)

| Badge | Category | How to Earn | Method |
|-------|----------|-------------|--------|
| Competitor | Competitions | Submit to any competition | `kaggle competitions submit` |
| Getting Started Competitor | Competitions | Submit to Getting Started comp | Submit to Titanic |
| Playground Competitor | Competitions | Submit to Playground comp | Find + submit to active playground |
| Community Competitor | Competitions | Submit to Community comp | Find + submit to active community |
| Code Submitter | Competitions | Code-based submission | Push notebook with competition_sources |
| Notebook Modeler | Competitions | Notebook that generates submission | Notebook + competition data |
| Competition Modeler | Competitions | Use model in competition notebook | Notebook + model + competition |

## Phase 3: Pipeline (~3 badges, 15-30 min)

| Badge | Category | How to Earn | Method |
|-------|----------|-------------|--------|
| Dataset Pipeline Creator | Datasets | Create dataset from notebook output | Push notebook → wait → create dataset from output |
| Model Pipeline Creator | Models | Create model from notebook output | Push notebook → wait → create model from output |
| R Markdown Coder | Notebooks | Push R Markdown notebook | `kaggle kernels push` (language=rmarkdown) |

## Phase 4: Browser-guided (~8 badges)

| Badge | Category | How to Earn | Method |
|-------|----------|-------------|--------|
| Stylish | Account | Fill out profile (bio, location) | User or explicitly authorized host agent |
| Vampire | Account | Switch to dark theme | User or explicitly authorized host agent |
| Bookmarker | Community | Bookmark content | User or explicitly authorized host agent |
| Collector | Community | Add item to collection | User or explicitly authorized host agent |
| GitHub Coder | Notebooks | Link GitHub repo to notebook | Manual (notebook settings) |
| Colab Coder | Notebooks | Open notebook in Colab | Manual (notebook menu) |
| Linked Dataset Creator | Datasets | Create URL-linked dataset | Manual (dataset creation UI) |
| Linked Model Creator | Models | Create externally-linked model | Manual (model creation UI) |

## Phase 5: Streaks (~4 badges, multi-day)

| Badge | Category | How to Earn | Duration |
|-------|----------|-------------|----------|
| 7-Day Login Streak | Account | Log in 7 consecutive days | 7 days |
| 30-Day Login Streak | Account | Log in 30 consecutive days | 30 days |
| Submission Streak | Competitions | Submit 7 consecutive days | 7 days |
| Super Submission Streak | Competitions | Submit 30 consecutive days | 30 days |

## Not supported by the collector workflow (~17 badges)

| Badge | Category | Why unsupported |
|-------|----------|-----------------|
| Contributor | Community | Requires progression tier (needs community engagement) |
| Expert | Community | Requires progression tier |
| Master | Community | Requires progression tier |
| Grandmaster | Community | Requires progression tier |
| Discussion Starter | Community | Requires upvoted discussion |
| Commentator | Community | Requires upvoted comment |
| Voter | Community | Requires voting on content |
| Sharer | Community | Requires external sharing |
| Course Completer | Community | Requires completing Kaggle Learn course |
| Certificate Earner | Community | Requires earning Kaggle Learn certificate |
| Competition Medal | Competitions | Requires earning a medal (performance-based) |
| Dataset Medal | Datasets | Requires earning a medal (community votes) |
| Notebook Medal | Notebooks | Requires earning a medal (community votes) |
| Team Player | Competitions | Requires joining a competition team |
| Competition Host | Competitions | Requires hosting a competition |
| Simulations Competitor | Competitions | Requires active Simulations competition |
| Featured Competitor | Competitions | Requires Featured competition (often $$$) |

## Summary

| Phase | Badges | Workflow support | Method |
|-------|--------|------------------|--------|
| 1 — Instant API | 16 | Yes | Automatic prerequisite actions |
| 2 — Competition | 7 | Yes | Automatic prerequisite actions |
| 3 — Pipeline | 3 | Yes | Automatic prerequisite actions |
| 4 — Browser-guided | 8 | Yes | Manual or explicitly authorized host agent |
| 5 — Streaks | 4 | Yes | Daily action helper; multi-day verification required |
| Not supported | 17 | No | — |
| **Total** | **55** | **38** | — |

An action reported as successful is not proof that a badge was awarded. Record
`verified` only after the badge is visible on the user's Kaggle profile.
