# exp266 PF ANCC / PF-Z multiseed stability audit

## 状態

- Route: `pf_beam`
- 状態: Kaggle CPU train v3完了、train-side multiseed stability audit採用
- 親: `exp072_exp063_full_replay_feature_cache`
- Kaggle kernel: `kentookumura/exp266-pf-ancc-pf-z-multiseed-stability-train` version 3
- inference / submission: 無効

## 仮説

`11d0f5ac`のPF ANCC / PF-Z優位がPF dynamicsの安定したmode追従なら、独立seedでも低RMSE、
HMM / likelihood-PFへのmargin、終端方向を高率に再現する。元seedだけの偶然なら、元seedは新規seed
分布の極端な下位tailに入り、同じ優位の再現率は低くなる。

## 結論

`11d0f5ac`のPF ANCC / PF-Z優位は単一seedの偶然ではない。新規63 seedで両手法ともHMM・
likelihood-PFへのstrong marginを100%再現し、RMSE 5 ft以下もPF ANCC 98.4%、PF-Z 100%だった。
PF ANCC元seedは分布中央値、PF-Z元seedはむしろ悪い側95.2 percentileである。

元seedstrong 53 wells全体では、両手法の過半数seedがstrongだったのは21 wells、80%以上再現は11 wells、
全seed再現は4 wellsだった。両手法で80%以上のseedがRMSE 5 ft以下だったのは`11d0f5ac`と
`fb0904bd`だけで、strong phenotypeは存在するが異質かつ一部seed選択バイアスを含む。

## 固定実験契約

- PF ANCC / PF-Z: exp072 exact kernel、各600 particles × 64 seeds。
- 全3,783,989 rows / 773 wells、seed 0 exact parity必須。
- PF dynamics 2 variants、LightGBM 0 config、fold 0、booster 0、GPU 0。
- true TVTは全pathと固定集約を生成した後の診断だけに使用。

## 検証方針

- seed 0のexp072全行exact parityを必須guardとする。
- 新規63 seedのwell別RMSE分布、元seed percentile、5/10 ft成功率、終端符号、HMM / likelihood-PF /
  exp226勝率を全773 wellsで測る。
- seed数1/4/8/16/32/64のmean / median / 10% trimmed meanを固定順序で比較する。
- strong phenotype 53 wellsとその他720 wellsを分け、raw条件、tail長、reference-method失敗度との関係を監査する。

## 所見

- runtime 12,482.144秒、seed 0 parityは両手法とも全行差0。
- 64 seed mean pooled RMSE: PF ANCC 14.493051 → 12.830319、PF-Z 17.788171 → 17.074522。
- PF ANCC 4 seed meanは13.126896で、64 seed改善量の約82%を回収した。
- 長いevaluation tailは誤差・seed分散を増やすが、strong発生を説明する単一raw thresholdはない。
- HMM / likelihood-PFの失敗度とstrong再現率の関係がraw特徴より強い。

詳細は`result.md`、実行履歴は`SESSION_NOTES.md`、機械可読値は`metrics.json`を参照する。

## 実行入口

- train: `exp266_pf_ancc_pf_z_multiseed_stability_audit_train.ipynb`
- inference: disabled notebook
- Kaggle kernel: `kentookumura/exp266-pf-ancc-pf-z-multiseed-stability-train`
