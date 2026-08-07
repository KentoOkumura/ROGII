# Type-well and horizontal well Overlap.

- archived_at: 2026-06-11T13:49:46Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703074

Topic #703074: Type-well and horizontal well Overlap.
  Author: Shrey Gandhi
  Posted: 2026-05-28 13:19:12.158000
  Votes: 0  Comments: 3

Looking at other discussions and the YT videos, geo-steering seems to be a GR overlap problem from the typewell to the horizontal well. 

But on looking at the overlaps of the wells, some wells have terrible overlaps. What do the domain experts use here to decide the TVT values?

Specially the segment between 15500 and 16000.
Example:
well pair_id = '000d7d20'

Comments:
├─ Chris Deotte (2026-05-28 14:53:51.270000) [+3]
│  Well 000d7d20 has great correlation between GR horizontal and vertical well:
  ├─ Shrey Gandhi (2026-05-28 15:14:25.547000) [+1]
  │  I see your plot is between TVT and GR. 
  │  Also, is vertical well same as typewell or ar these the values before PS.
  │  Great analysis btw, could be used somewhere.
    ├─ Chris Deotte (2026-05-28 15:20:22.300000) [+0]
    │  What i call "vertical well" is typewell.csv. And what i call "horizontal well" is the TVT before and after PS.
    │  
    │  I make this plot with df.groupby('TVT').GR.mean() applied to the two dataframes, hori...
