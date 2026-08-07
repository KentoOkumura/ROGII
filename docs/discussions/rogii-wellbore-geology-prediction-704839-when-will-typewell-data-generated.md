# When will Typewell data generated?

- archived_at: 2026-06-11T13:48:57Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704839

Topic #704839: When will Typewell data generated?
  Author: doteeee
  Posted: 2026-06-06 14:27:33.907000
  Votes: 2  Comments: 5

I understood, logs from Horizontal well got generated from while drilling. As the drill starts vertically and move horizontally, we can get information regarding geological formation across.
Logs generated -&gt; coordinate(x,y,z)  and GR , TVT 

But  i am not able to understand when will Typewell information will get generated. Require some clarification regarding this, to understand if GR, TVT from typewell can be better used relate to TVT of horizontal well.

Thanks

Comments:
├─ PC Jimmmy (2026-06-06 16:22:39.780000) [+2]
│  Vertical type wells exist before the drilling starts - so full data set should be present at test when the scoring run starts.  
│  
│  There are a limited number of different vertical holes in the train...
  ├─ doteeee (2026-06-06 17:30:21.193000) [+0]
  │  thanks, @pcjimmmy. so the typewell are the vertical drilling happened somewhere near to horizontal well even before work starts to estimate the geology. is my understanding correct?
    ├─ PC Jimmmy (2026-06-07 02:05:31.240000) [+1]
    │  Yes - some of them also would be capable of oil or gas if the layer thick enough or perhaps some lease issues preventing the folks from going horizontal, but generally low ROI to drill them.  So my...
    ├─ PC Jimmmy (2026-06-07 02:11:11.293000) [+2]
    │  "pseudo-typewells" is the term used rather than synthetic.
    ├─ doteeee (2026-06-07 04:02:20.350000) [+0]
    │  thank @pcjimmmy , will go through refered post
