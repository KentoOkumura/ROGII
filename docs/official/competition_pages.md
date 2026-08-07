# ROGII 公式ページ要約

取得日: 2026-05-27  
取得元: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction  
取得方法: Kaggle CLI/API (`kaggle competitions pages ... --content`)

## Description

ROGII - Wellbore Geology Prediction は、horizontal wellbore に沿って遭遇する geology を予測するコンペです。掘削中の地質解釈を自動化し、より正確な well placement、資源浪費の削減、安全性向上に寄与することが目的です。

## Data

- train/test は well ごとに 8 文字 hash の `WELLNAME` で管理される。
- train の各 well は horizontal CSV、typewell CSV、PNG を持つ。
- test の各 well は horizontal CSV と typewell CSV を持つ。
- test の公開ファイルは authoring 用の少数例で、Kaggle 提出実行時には hidden test に差し替えられる。
- 予測対象は evaluation zone の `TVT`。提出列では小文字の `tvt`。
- `TVT_input` は既知区間の `TVT` コピーで、evaluation zone は NaN。

## Timeline

- 2026-05-05: Start Date
- 2026-07-29 23:59 UTC: Entry Deadline
- 2026-07-29 23:59 UTC: Team Merger Deadline
- 2026-08-05 23:59 UTC: Final Submission Deadline

## Code Requirements

- Notebook-only submission。
- CPU Notebook と GPU Notebook はどちらも 9 hours 以内。
- Internet access disabled。
- Freely & publicly available external data と pre-trained models は許可。
- 提出ファイル名は `submission.csv`。

## Prizes

- 1st Place: $25,000
- 2nd Place: $13,000
- 3rd Place: $7,000
- 4th Place: $5,000

## Rules Notes

- team 外の private code/data sharing は不可。
- 公開コード共有は Kaggle の competition forum/notebook 上で全参加者に公開する。
- validation/test records の hand labeling や human prediction は不可。
- 同点時は先に提出された submission が上位。
- 最大チームサイズは 5、1 日の最大提出数は 5。
