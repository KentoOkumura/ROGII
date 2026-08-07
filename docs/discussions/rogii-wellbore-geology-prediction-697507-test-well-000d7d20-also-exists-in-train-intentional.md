# Test well 000d7d20 also exists in train/  (intentional?)

- archived_at: 2026-06-11T13:50:43Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697507

Topic #697507: Test well 000d7d20 also exists in train/  (intentional?)
  Author: Maher el Ouahabi
  Posted: 2026-05-06 12:49:35.680000
  Votes: 2  Comments: 2

Hi @igorkuvaev @sergeyalyaev,

While exploring the data I noticed that the well_id 000d7d20 appears
both in train/ and in test/. The horizontal trajectory (MD, X, Y, Z)
matches row-for-row across the two folders for all 5,278 rows; the GR
log differs in ~43% of rows (likely intentional noise).

Two questions:



Is the visible test/000d7d20 a duplicate of train/000d7d20 by design
(e.g., a sanity sample for kernel debugging), or is it an accidental
carry-over that will be re-mounted as a new well at hidden-test
scoring time?

The other two visible test wells (00bbac68, 00e12e8b) do not appear
in train/. Is the hidden test set composed of mostly new wells, or
is some overlap with train expected (consistent with synchronous
reruns + held-out split)?


Asking because this directly affects whether the right strategy is to
condition predictions on train/&lt;well_id&gt; existence, or to ignore that
path entirely. Want to make sure we're modeling the right problem.

Thanks!

Comments:
├─ Ryan Holbrook (2026-05-06 14:00:02.230000) [+5]
│  Hi @maherelouahabi,
│  
│  The wells you see in test/ are are just example data to help you author your submissions. When you submit your notebook, the example test data will be replaced with the actual ...
├─ lingyu07 (2026-05-06 12:51:29.767000) [+0]
│  3 wells :
│  
│  000d7d20: 3836 hidden rows
│  
│  00bbac68: 6014 hidden rows
│  
│  00e12e8b: 4301 hidden rows
