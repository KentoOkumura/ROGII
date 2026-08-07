# Some question

- archived_at: 2026-06-11T13:50:44Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697300

Topic #697300: Some question
  Author: YePing Lin
  Posted: 2026-05-05 18:18:55.055000
  Votes: 3  Comments: 3

Hello everyone, may I ask how to perform data augmentation in this kind of competition? Or rather, is data augmentation reasonable (because I'm not sure if it will work)? This is my first time participating in this kind of competition, so I don't know much about it. Thank you to all the experts for your guidance.

Comments:
├─ hengck23 (2026-05-06 12:01:51.690000) [+2]
│  interpolate nearby wells? you have to imagine you are subsampling from a big 3d geological structure. augmentation here means creating more samples (possibly virtual ones) from this big structure
├─ Abdessamed Zetroni (2026-05-05 19:25:21.987000) [+1]
│  Data augmentation has limited use here because of the nature of the problem.
│  
│  What could work: adding small noise to GR logs, or randomly shortening the known zone during training to simulate diffe...
  ├─ hengck23 (2026-05-06 12:06:47.123000) [+4]
  │  warp the GR data for both reference and input  … like linear elastic transform.
  │  
  │  make aligned pairs reference and input to decide what kinds of noise, deformation to add
