# 要件

## 依頼

exp063 は公開 Pixiux notebook から LightGBM 部分を strict replay して提出した。今回は LightGBM booster を通さず、同じ公開 replay feature generation から得られる Pixiux PF/Beam / likelihood-PF 予測値をそのまま `submission.csv` にして提出候補を作る。

## 制約

- Route: `pf_beam`
- 新しい学習モデルは作らない。
- 既定の提出列は `likpf_mean` とする。
- static visible override、final public notebook blend、projection postprocess、CatBoost、Ridge stack、hidden-specific branch は入れない。
- exp063 `lgb_mean` と exp027 PF route anchor との差分比較を可能な範囲で保存する。

## 受け入れ基準

- `experiments/exp069_pixiux_pf_beam_direct_submit_audit/` に notebook、config、実装、記録が揃う。
- inference notebook は Kaggle 上で `submission.csv` を生成できる。
- 候補予測、候補別要約、reference submission との差分、tracker features を生成物として保存する。
- `validate_experiment.py`、Python compile、notebook JSON validation が通る。
