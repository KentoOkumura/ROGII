# Issue after submission file !! Need Help !

- archived_at: 2026-06-11T13:48:41Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699110

Topic #699110: Issue after submission file !! Need Help !
  Author: Yash Kumar Saini
  Posted: 2026-05-12 19:02:20.820000
  Votes: 1  Comments: 1

Hello, 
Why it's failed after this, can anyone help me on this ?

`

77.5s    40                id           tvt
77.5s    41  0  000d7d20_0000  11244.527899
77.5s    42  1  000d7d20_0001  11244.527899
77.5s    43  2  000d7d20_0002  11244.510133
77.5s    44  3  000d7d20_0003  11244.638217
77.5s    45  4  000d7d20_0004  11244.621056
77.5s    46  submission.csv created successfully!
80.5s    47  /usr/local/lib/python3.12/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '|'
80.5s    48    cells[i][c] = re.sub('\\|', '|', cell)
80.6s    49  /usr/local/lib/python3.12/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '_'
80.6s    50    text = re.sub(r'_', '_', text) # Escape underscores in display text

`

Comments:
├─ Aly Ayman (2026-06-08 20:25:14.890000) [+-1]
│  Hi! Good news first: based on what you've pasted, your code actually ran fine. The lines you're seeing are not the cause of the failure:
│  
│  
│  
│  submission.csv created successfully! confirms your file w...
