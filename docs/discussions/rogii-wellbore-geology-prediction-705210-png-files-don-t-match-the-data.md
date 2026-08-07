# PNG files don't match the data

- archived_at: 2026-06-11T13:48:22Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/705210

Topic #705210: PNG files don't match the data
  Author: MichaelB
  Posted: 2026-06-09 02:41:24.544000
  Votes: 9  Comments: 11

Others have observed that the PNG (and PPTX) show meters instead of feet (https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700871 and https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703534) but that's not the only difference I'm seeing.

1) The PNG files show 57 unique Typewell numbers, but there are actually 752 unique {well}__typewell.csv files in train. To take just one example, wells 0dd99dc5 and 028d7b28  have typewell.csv files of lengths 1992 and 1774 respectively (and different values in them), but their PNGs both show Typewell20001. Most of the images showing the same typewell are not in fact the same typewell in the data. Additionally, 21 wells shown as having different typewells in the PNGs actually have duplicate typewell.csv files in the data. Total mismatch.

2) The "TVT plot (last 200 FT)" in the PNGs do not in fact correspond to the last 200 feet (or the last 200 rows, or anything else I can make sense of) in the well data we've been given. For example, for well 101a1281a the provided chart goes from like 11070 to 11270. Or for well 0a57a29c, the provided chart goes from 11015 to 11210 instead of 11195.40.

3) The charts show different values than what we have. I wrote some code to extract all the structured data from the PNG files (including Typewell and Azimuth), and duplicate the PNG visualizations as best I could. To take just one example, compare the attached image for well 0a57a29c with the 0a57a29c.png we've been provided. Ignoring minor cosmetic differences (fonts, legend placement, the omitted "GR range: 0-200" label) there are still differences. For example, the TVT plot in the original PNG we were given keeps going beyond the data in our 0a57a29c__horizontal_well.png, and the GR values for the horizontal well in the last 200 ft plot don't seem to line up either. The initial dips in the Well Path Projection chart are not quite the same (I'm using the same formula for Horizontal Distance that Dmitry Stadnik computed in https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700871)

4) All the PNGs show "GR range: 0-200" but 2047 GR values in the provided horizontal well data files are &gt; 200.

Comments:
├─ Navneet (2026-06-11 08:08:46.780000) [+-1]
│  Thanks for the PNG files info @brundage
├─ PC Jimmmy (2026-06-09 15:21:06.983000) [+2]
│  Did you look at data values for the two wells in your first comment - 0dd99dc5 and 028d7b28 - I am betting that with a little lag (109 feet or so)  you can see they are the same hole.  
│  
│  I also wro...
  ├─ PC Jimmmy (2026-06-09 15:48:40.457000) [+1]
  ├─ PC Jimmmy (2026-06-09 16:58:55.380000) [+0]
    ├─ PC Jimmmy (2026-06-09 17:07:38.663000) [+0]
    │  If you plot the wells that tie to 20001 you might start getting nervous about the value of the type well depending on it's location.  For sure in the upper right corner of the plot there would be Z...
    ├─ PC Jimmmy (2026-06-09 17:11:27.767000) [+0]
    ├─ PC Jimmmy (2026-06-09 17:13:41.297000) [+1]
    │  This is my personal favorite - no way in heck that all these wells were drilled using the 20016 typewell.  IMO most would have been drilled using data from an older adjacent horizontal.
    │  
    │  The piece ...
    ├─ MichaelB (2026-06-09 20:11:35.530000) [+0]
    │  Thanks, these are good insights. I'll dig in more. I was just doing a naive comparison.
    │  
    │  
    │    
    │  I actually think that 57 typewells are more than what the real world drilling saw for this field and tha...
    ├─ Igor Kuvaev (2026-06-10 00:40:33.120000) [+4]
    │  Great discussion, guys!
    │  
    │  All of the data is real-there's no synthetic data here.
    │  
    │  The selection of the type well for lateral steering is somewhat subjective and depends on the geologist. To make th...
    ├─ Sandy (2026-06-10 13:01:57.043000) [+1]
    │  Why do you say that it's not possible to drill multiple horizontal wells using a specific typewell?
    │  In my limited knowledge, a typewell could be an exploratory well (I'm not aware of the actual det...
├─ OpPrime (2026-06-09 08:49:22.020000) [+1]
│  Good to know others are looking at the problem from the same lens, we started rendering from real data to avoid this.
