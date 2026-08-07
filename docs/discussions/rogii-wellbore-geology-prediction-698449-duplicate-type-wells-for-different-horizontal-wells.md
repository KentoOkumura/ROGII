# Duplicate type wells for different horizontal wells

- archived_at: 2026-06-11T13:50:38Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698449

Topic #698449: Duplicate type wells for different horizontal wells
  Author: RMorrison
  Posted: 2026-05-09 23:30:08.615000
  Votes: 18  Comments: 3

When doing a quick EDA, I noticed there are a few type well files that are exactly the same but assigned to different horizontal wells. In some cases this makes sense (wells that are oriented adjacent or on top of each other), but in other cases the wells are quite distant from each other. Why would wells some distance apart be using exactly the same type well? Is this intentional?

Here are the well pairs with matching type wells, and a picture of how they are oriented compared to other wells with different type wells.

duplicate_groups = [
    ['02e7fe5a', '10b89021', '3417285d', '6ae68655', '7993a768', 'bc4381e2', 'ecdab904', 'f021b650', 'f49fdea3', 'f88ddb26'],
    ['071d7b45', '4463446c'],
    ['25939962', '8050c789'],
    ['2f8e53c3', '91db7070'],
    ['75cd5f11', 'be83e781'],
    ['7b38844c', 'ed6e6e54'],
    ['89f1085d', 'aed44918'],
    ['8b95d6d1', 'a2e8e7f6'],
    ['a4f989c2', 'd011f41b'],
    ['add9c322', 'cbe62450'],
    ['b977be4a', 'c908edd0'],
    ['cd7f1687', 'fcfcc902'],
    ['f321a31c', 'fa667be2'],
]

Comments:
├─ Igor Kuvaev (2026-05-10 03:57:30.960000) [+7]
│  Good catch!
│  
│  Typically typewell selection is a manual process, the geologist picks the one that is close to the lateral and available at the time.
│  Wells in the project were drilled over 10 years an...
├─ Matthew Degtyar (2026-05-10 15:45:31.637000) [+0]
│  Very clutch
├─ PC Jimmmy (2026-05-10 00:35:24.093000) [+0]
│  Nice catch!
