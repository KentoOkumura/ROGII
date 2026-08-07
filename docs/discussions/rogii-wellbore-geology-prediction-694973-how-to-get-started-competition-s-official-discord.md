# How to get started + Competition's Official Discord

- archived_at: 2026-06-11T13:48:17Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/694973

Topic #694973: How to get started + Competition's Official Discord
  Author: María Cruz
  Posted: 2026-04-27 21:33:19.370000
  Votes: 10  Comments: 18

Information for newbies

New to machine learning and data science? No question is too basic or too simple. Feel free to start your own thread, or use this thread as a place to post any first-timer clarifying questions for the Kaggle community to help you with!

New to Kaggle? Take a look at a few videos to learn a bit more about site etiquette, Kaggle lingo, and how to enter a competition using Kaggle Notebooks. Publish and share your models on Kaggle Models!

Looking for a team? Express your interest in joining a team through our Team Up feature.

Remember: Kaggle is for everyone. Whether you're teaming up or sharing tips in the competition forum, we expect everyone to follow our Kaggle community guidelines.

Competition's Official Discord

In addition to this competition forum, you can continue the discussion in our official Kaggle Discord Server here:

discord.gg/kaggle

The Discord is a great place to ask getting started questions, chat about the nuances of this competition, and connect with potential team mates. Learn more about Discord at our announcement here. Here are a few things to keep in mind though:

1. Discord Competition Channels are 'Public' - Don't Share Private Information

Discord channels for specific competitions are considered 'public' spaces where you are allowed to talk about competition details. Please remember that private sharing of competition code or data outside of your team is, as always, not permitted. Code sharing must always be done publicly through the Kaggle forums/notebooks.

2. Discord Competition Channels are Not Monitored by Staff - Keep Important Information on the Kaggle Forums

Kaggle Staff and Hosts running competitions will not monitor Discord or be available to answer questions in Discord. This is intended to be a more casual space to discuss competitions and help each other. Please keep important questions, insights, writeups, and other valuable conversation on the Kaggle forums. 

Happy modeling!

Comments:
├─ Luis Diambra (2026-05-20 19:54:01.800000) [+1]
│  Hi Team. This is my first time participating in Kaggle. I've submitted file, but but I'm getting a "submission score error" message, even though the file has the correct format. When I check the lo...
  ├─ PC Jimmmy (2026-05-20 23:27:07.833000) [+1]
  │  The log files of no value for a submission error, assuming the the save and run worked ok.
  │  
  │  The best way to get help is to make your notebook public and than add a link here in the discussion post....
    ├─ Luis Diambra (2026-05-21 13:14:28.377000) [+0]
    │  Thank you very much Jimmy, I made public my notebook:
    │  https://www.kaggle.com/code/luisdiambra/notebookb7f2cbd34e
    │  I am becoming crazy; now the systems says "This Competition requires a submission fi...
    ├─ PC Jimmmy (2026-05-21 14:12:09.900000) [+1]
    │  playing with it now - will hollar if I figure out anything
    ├─ PC Jimmmy (2026-05-21 14:35:27.470000) [+0]
    │  I agree the file exists.  But you need to remember that the submission.csv file you see is generated using only a couple of wells.  The real submission works on a larger count of files.  Pretty com...
    ├─ PC Jimmmy (2026-05-21 14:37:57.297000) [+0]
    │  Found this issue - trying to revise the code - the error output when we do a submission is always pretty NOT useful :)  Past practice from years ago - skilled folks could use error messages to "che...
    ├─ PC Jimmmy (2026-05-21 14:41:40.440000) [+0]
    │  The first submission I made with no changes - this is the error output that you should share when seeking help rather than just submission score error.
    │  
    │  *Your notebook generated a submission file w...
    ├─ PC Jimmmy (2026-05-21 14:50:57.787000) [+0]
    │  My attempt to fix the first issue did not work - I got the exact same full error message.
    │  
    │  I have run out of submissions for the day - will look again later when a new kaggle day starts.  Found a s...
    ├─ PC Jimmmy (2026-05-21 14:57:53.677000) [+1]
    │  I made a mistake in cell 3 attempting to fix the index issue.  The link above should take you to a "better fix" :)  Will do the submission in around 12 hours.
    ├─  (2026-05-21 15:48:49.390000) [+0]
    ├─ Luis Diambra (2026-05-21 16:01:09.203000) [+0]
    │  This comment is valuable; I've never seen it before. I don't know how you got it or how to access it. Even so, it wasn't helpful in determining the problem because the submission files I sent have ...
    ├─ Luis Diambra (2026-05-21 16:03:48.140000) [+0]
    │  Many thanks for the comments. That is my first doubt. There are 3 wells in the test set. Should the submission file only contain the prediction for these 3 wells, as indicated in the sample_submiss...
    ├─ PC Jimmmy (2026-05-22 00:32:03.553000) [+0]
    │  If you click on the words Submission Scoring Error just below the notebook name on the submissions tab it would take you to the additional comments.
    │  
    │  What you can only see are the 3 wells in the te...
    ├─ PC Jimmmy (2026-05-22 00:39:20.673000) [+1]
    │  My attempt has same error.  Don't think I can figure it out.  Sorry
  ├─ José Luiz Luna-Xavier (2026-05-21 16:51:13.620000) [+0]
  │  Hi Luis, I checked your notebook and logs. The notebook seems to run until the end: the logs do not show a fatal Python crash, only pandas warnings. The issue is most likely with the generated subm...
    ├─ Luis Diambra (2026-05-21 17:54:22.257000) [+0]
    │  Thank you so much for looking at my code. I've already addressed the NAN in GR, filling in the gaps. I've added a quality control check to the end of my notebook for my submission.csv file, checkin...
    ├─ José Luiz Luna-Xavier (2026-05-21 18:35:26.737000) [+0]
    │  I checked the CSV file you shared. The file itself looks structurally valid: it has exactly the columns id,tvt, 14,151 rows, 14,151 unique IDs, no duplicates, no NaN values, no infinite values, and...
    ├─ Luis Diambra (2026-05-21 19:12:51.697000) [+0]
    │  Thanks for your efforts in helping me. Unfortunately, it still gives the same problem. My output is generated in the /kaggle/working/ subdir as submission.csv. I tested again with this new public n...
