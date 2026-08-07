# Defintion of tvt

- archived_at: 2026-06-11T13:49:59Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698282

Topic #698282: Defintion of tvt
  Author: Moonimonster
  Posted: 2026-05-09 06:29:46.065000
  Votes: 14  Comments: 13

Hope y'all enjoying this competition!

I was wondering what you all defined this 'tvt', the target.
It seams like it means just literal 'depth' as described in typewell.
but I don't really think it is true meaning of tvt.. I searched up on internet, and I found out it means thickness of 'something'., but I'm not really sure what 'something' is in this competition.

Pretty sure, it would mean something more than 'depth', cause.. what would be the meaning of finding tvt where we have z right? 
I'm still working on this issue.
Hope some of you can share ideas about it. Thank you!

Comments:
├─ Tom (2026-05-17 02:46:29.920000) [+17]
│  Here are some cheetsheets I made. Hope this can help you understand.
  ├─ Moonimonster (2026-05-19 06:40:58.090000) [+0]
  │  Thank you!!😀
├─ PatrickAIForFun (2026-05-09 11:14:22.527000) [+2]
│  Please correct me if I am wrong here.
│  Based on my EDA and other discussions it seems that TVT in this case is the vertical distance to a virtual/imaginery reference line. This aligns somewhat with ...
  ├─ Igor Kuvaev (2026-05-10 04:02:10.810000) [+5]
  │  PatrickAIForFun
  │  you are correct - TVT is vertical distance to a virtual/imaginery reference line
  │  TVT=0 is referencing the ground level - this may not be the case. 
  │  Most importantly TVT in the later...
    ├─ Evdilos_Ikaria (2026-05-10 15:34:29.527000) [+0]
    │  Is it possible to give us a small sketch of TVT , Z  in a horizontal drill
    │  Thanks
    ├─ PatrickAIForFun (2026-05-11 19:24:51.663000) [+0]
    │  Dear @igorakuvaev 
    │  Thank you for the info and confirmation - this helps a lot.
    │  A quick follow up question:
    │  Are the TVT scales aligned across lateral and typewells? I.e. would it also be possible to...
    ├─ Igor Kuvaev (2026-05-14 06:10:16.320000) [+3]
    │  TVT is not aligned between different wells, it is only aligned between well and typewell
    ├─ Dmitry Stadnik (2026-05-18 11:13:13.573000) [+0]
    │  are well and typewell different physical wells drilled close to each other?
    ├─ Brian Lynch (2026-05-24 17:59:57.370000) [+0]
    │  This conflicts with what I’ve found online, but of course as the host what you say should be taken as truth for the competition!
├─ hengck23 (2026-05-12 03:08:39.847000) [+0]
│  this is chatgpt interpretation (from https://github.com/rogii-com/Python-SDK/tree/examples/tvt_rop_heatmap)
│  
│  XYZ = real spatial trajectory
│  GR = real log along trajectory
│  TVT = interpreted coordinat...
  ├─ Moonimonster (2026-05-12 05:51:03.723000) [+0]
  │  Thank you for sharing! 😀
├─ PC Jimmmy (2026-05-09 15:32:47.693000) [+0]
│  Using one of the AI's to help with coding I realized that it was confused about the meaning of tvt also :)
│  
│  A general meaning could be that it was the thickness of the geological layer.  That's the...
  ├─ Moonimonster (2026-05-11 23:44:43.967000) [+0]
  │  Thank you for sharing your idea!😀
