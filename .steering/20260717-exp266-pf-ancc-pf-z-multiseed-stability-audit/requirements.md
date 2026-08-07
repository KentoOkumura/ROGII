# 要件

## 依頼

`11d0f5ac`で良好だったPF ANCCとPF-Zはexp072の単一stable seed出力であり、偶然の可能性がある。
seedを増やし、同wellでの再現性、他wellでの同型現象、発生条件を全wellで検証する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親exp072のPF ANCC / PF-Zアルゴリズムと600 particlesを固定し、変更はseed反復数だけとする。
- 全3,783,989 rows / 773 wellsを対象とし、対象wellを結果で絞らない。
- 各方式64 seedsとし、seed index 0はexp072元seed、1〜63はimmutable keyからSHA256で生成する。
- true TVT、target、error、oracleは候補path生成とseed集約へ使用せず、生成後の診断だけに使用する。
- CPU notebook 1本、PF dynamics 2 variants、LightGBM config 0、fold 0、booster 0、GPU 0、親/control再学習0、inference / submission 0とする。

## 受け入れ基準

- seed index 0のPF ANCC / PF-Zがexp072保存列と全3,783,989行でexact parityを満たす。未達ならmultiseed結果を採用せず停止する。
- per-well × algorithm × seedのRMSE、MAE、bias、終端誤差、終端符号、距離帯別RMSEを保存する。
- well × algorithmについて元seed percentile、seed分布q05/q10/q25/q50/q75/q90/q95、RMSE 5/10 ft以内率とWilson 95%区間、exp226/HMM/likPF比較率を保存する。
- nested seed count `1/4/8/16/32/64`でmean、median、10% trimmed meanのpath精度と収束を保存する。
- `11d0f5ac`、先行strong phenotype 53 wells、その他wellで再現率とseed不安定性の発生条件を比較できる。
- Kaggle kernel version、入力SHA、config/source/notebook SHA、gzip生成物のdecompressed content SHAを記録する。診断実験なのでmodel/submission SHAは対象外と明記する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
