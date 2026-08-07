# Horizontal distance along azimuth is wrong, or I am wrong?

- archived_at: 2026-06-11T13:50:11Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700871

Topic #700871: Horizontal distance along azimuth is wrong, or I am wrong?
  Author: Dmitry Stadnik
  Posted: 2026-05-18 09:37:55.943000
  Votes: 3  Comments: 2

I have 2 questions:



Why is it in meters on png and pptx while every other length/depth is in ft?

I'm trying to calculate departure (horizontal distance, D) from MD (measured depth) and TD (true depth).
The formula I'm using is:
D = sqrt(MD ^ 2 - TD ^ 2).
For the well 000d7d20 I see that D is in range 7000-14000 ft, while on png it's in range 0-5000 m (lower left chart). This is impossible: 5000 m is 16404.2 ft, which is very close to MDmax (total measured depth) of that well (16744 ft) - if that would be the case the well should go horizontal right from the surface.
Is my calculation incorrect?


The first image is my chart, the 2nd is the one from provided png.

Comments:
├─ Dmitry Stadnik (2026-05-19 05:10:42.703000) [+1]
│  My calculation was indeed wrong - the departure (horizontal projection of well's trajectory) must be calculated as cumulative sum of departures at each datapoint:
│  
│  well_data['dMD'] = well_data['MD'...
├─  (2026-05-18 15:50:20.540000) [+0]
