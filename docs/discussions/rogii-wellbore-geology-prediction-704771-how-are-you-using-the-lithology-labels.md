# How are you using the lithology labels?

- archived_at: 2026-06-11T13:49:04Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704771

Topic #704771: How are you using the lithology labels?
  Author: yiyu0716
  Posted: 2026-06-06 03:06:11.597000
  Votes: 3  Comments: 1

Hi everyone,

I have been trying to make use of the lithology / Geology labels, but so far I have not found a reliable way to turn them into a clear validation improvement.

They seem useful for understanding the typewell structure and the geological context, but in my experiments they have not yet worked well as direct features or matching signals.

Has anyone found a good way to use these labels? For example, are you using them for segmentation, filtering, alignment, post-processing, or only for visualization / interpretation?

Any suggestions would be appreciated.😀

Comments:
├─ Tucker Arrants (2026-06-06 05:53:01.670000) [+2]
│  The obvious thing to try is using them as an auxiliary training task, but so far it has provided zero benefit in my pipeline.
