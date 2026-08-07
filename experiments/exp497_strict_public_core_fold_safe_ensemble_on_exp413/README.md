# exp497_strict_public_core_fold_safe_ensemble_on_exp413

## 状態

- ルート: ensemble
- 状態: Stage E promotion gate FAILを維持、保存model hidden-safe inference version 2はKaggleで完了・output検証PASS
- CV: exp497 cross-fit blend 7.874488 / exp413 7.884803（-0.010315 ft）
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-08-01
- 親実験: exp413_scale5_likpf_full_replacement_on_exp335

## 仮説

Public LB固有のwell補正を除いた公開pipelineの中核を独立OOF trajectoryとして
再学習すれば、同じfinal370上でmodel familyを増やしたexp494よりexp413との
誤差相関が低くなり、constant convex blendで安全に改善できる。

## 変更点

- 公開pipelineの物理候補、well形状selector概念、SP45 residual ML、U投影、独立learned trajectory、warmup/smoothingを残す。
- 公開固定selector閾値/mapと内部weightは直接使わず、outer-train inner OOFだけでfitする。
- exp413は保存OOFだけを使い、public-core完成後のmeta-fold blendまで入力しない。
- fixed public well ID、Q0522/A27、同一well contact、visible-prefix overlay、precomputed submission/public outputを除外する。

## 検証方針

- Fold: exp413 outer 5 / public-core inner 4 / ensemble meta 5
- Group: well
- Score rows: TVT_input missing suffix rows、3,783,989 rows / 773 wells
- Leakage check: outer-valid wellをmodel、Ridge、selector、weight、spatial imputer poolから除外する。
- Primary gate: exp413比0.03 ft以上、5/5 folds、全scope nonworse、by-well p95/worst各+0.25 ft以下、全meta-foldでpublic-core weight正。

## 実行量（設計値）

- scientific variant: 1
- ML branch: 2
- 各branch: LightGBM 3 + CatBoost 2、outer 5 × inner 4
- LightGBM / CatBoost / total booster: 120 / 80 / 200
- Ridge: 10
- exp413 control/selector/signed/TVT再学習: 0 / 0 / 0 / 0
- LikPF: selector / learned各773 seed-bank、合計197,888 seed-well / 98,944,000 particle starts
- Beam: selector 14 × 773 + learned 7 × 773 = 16,233 well-config runs
- PF ANCC / PF Z: 各773 well runs、NCC: 2,319 well-window runs
- SP45 / learned feature: 195 / 205列、2面float32同時保持は約6.05 GBのため禁止

## 実行入口

正規`train.ipynb`はsharded Kaggle実行runbookとして採用済み。実処理はJupytext起点の
`pfbeam_features_fold0..4`、`train_fold0..4`、`train_aggregate` Notebookで行う。Stage P/Eは
Kaggle CPU、Stage Mは各fold 40 boostersをKaggle GPUで実行済み。Stage I version 4で保存した
strict public-core 40 boostersとRidge 2を、正規`inference.ipynb`が学習0で読み込む。exp413も
75 saved boostersをdynamic hidden-safe runtimeで読み、固定weightでblendする。Kaggle outputには
`submission.csv`を生成するが、外部competition submitは行わない。Colabは使わない。

## 結果

| メトリック | 値 |
| --- | --- |
| Stage P/M/E実装 | 完了、contract test 23件PASS |
| Stage M fold0 | 757,738 rows / 155 wells / 40 boosters / 12,396.747秒 |
| fold0 strict public-core RMSE | 9.281962 |
| Stage M fold1 | 756,650 rows / 155 wells / 40 boosters / 15,179.320秒 |
| fold1 strict public-core RMSE | 8.413250 |
| Stage M fold2 | 756,255 rows / 154 wells / 40 boosters / 14,041.621秒 |
| fold2 strict public-core RMSE | 8.467883 |
| Stage M fold3 | 757,101 rows / 155 wells / 40 boosters / 16,074.250秒 |
| fold3 strict public-core RMSE | 8.732576 |
| Stage M fold4 | 756,245 rows / 154 wells / 40 boosters / 14,201.086秒 |
| fold4 strict public-core RMSE | 10.121002 |
| exp413 CV | 7.884803 |
| exp497 cross-fit blend CV | 7.874488 |
| exp497 − exp413 | -0.010315 ft |
| nonworse folds | 3 / 5 |
| hidden-like spatial / typewell-purged delta | +0.105138 / +0.097410 ft |
| by-well delta p95 / worst | +0.700720 / +7.541588 ft |
| promotion gate | FAIL |
| selected prediction | exp413 OOF |
| Stage I version 4 | COMPLETE / 14,151 rows / 3 wells / 9,204.737秒 |
| Stage I fit | LightGBM 24 + CatBoost 16 / Ridge 2 / exp413再学習・再推論0 |
| Stage I prediction contract | sample ID順序一致 / 重複・欠損・非有限値0 |
| Stage I提出生成 | なし |
| Stage Iモデル重み | 40件保存・検証PASS（LightGBM 24 / CatBoost 16、335,918,672 bytes） |
| Stage I reload parity | 最大絶対差0.0（許容値1e-5） |
| Stage I model-set SHA | `dcc2166f4bd5731364efe0b3fb848a46cf87f8133cbe78890658a1062c604626` |
| Stage I Ridge weights SHA | `34aa73067d6e67b98eb72c40035b5065d6721674af89982a1089f1d803a6c727` |
| Stage I v4 prediction SHA | `6abd8b1d2c73d88cd8d8cfa0863cc9d08e89dbd97d1d7892d278c0d23e83f98e` |
| Saved-model inference | version 2 Kaggle T4 `COMPLETE`、output検証PASS |
| Saved-model inference読込 | exp497 40 boosters + Ridge 2 / exp413 75 boosters |
| Saved-model inference kernel | `kentookumura/exp497-strict-public-core-saved-inference` / id_no `129666751` |
| Saved-model inference fit | exp497 0 / exp413 0 / weight refit 0 |
| Saved-model inference runtime | kernel本体約684.077秒 / exp413 391.418秒 / strict特徴231.471秒 / 保存model推論33.509秒 |
| Saved-model inference parity | strict 0.001281 ≤ 0.002 / blend 0.014195 ≤ 0.020 ft、PASS |
| Saved-model inference output | 14,151行、sample ID順一致、重複・欠損・非有限値0、submit-check FAIL/WARN 0 |
| Saved-model inference submission SHA | `04ca2e2f80f45bced1e22bd68a58002b4cb7c7e5b19510932375cdccafa6680a` |
| 外部competition submit | 未実行 |
| Hidden対応 | dynamic sample ID + dynamic exp413、public-test固定sidecar禁止 |
| Public LB | 未提出 |
| Private LB | - |

## 所見

参照sourceのJupytext変換SHA `88c7b99e...5454`、必要symbol 17件、SP45 195列、
learned 205列、PF/Beam実行量を固定した。Stage P/M/Eの技術契約はPASSし、5 meta-fold weightも
すべて正だった。一方、pooled gainは0.010315 ftに留まり、fold 0/4、hidden-like 2面、
by-well tailを悪化させたため、全AND gateに従って不採用とした。

## リスク / 注意

- 200 boostersと2系統128-seed PFを含む高コスト設計である。
- exp494はCV 5/5改善でもLBが悪化したため、pooled平均だけで昇格しない。
- public sourceとのbyte parityは目的にせず、stable per-well seedとfold safetyを優先する。

## 次

selected final anchorはexp413を維持する。Stage I version 4の保存modelを読むhidden-safe inference
候補をJupytext起点で実装した。raw hidden testからstrict public-coreとexp413を動的再生成し、
exp497/exp413とも学習0、固定weightだけを適用する。正規inference Notebookへ採用し、76 support
filesのpackage readback、remote T4 / internet off / 13 kernel sourcesを検証してKaggle version 1を
pushした。version 1はdynamic exp413、strict特徴、全保存model推論後のvisible parityで`ERROR`。
OOM・入力欠落・model破損ではなく、外部submitは0。strict保存modelとdynamic exp413/blendの監査を
分離し、中間exp413 submissionを隔離するversion 2を同一T4 kernelで完了した。strict / blend
visible parityは`0.001281 / 0.014195 ft`で各許容`0.002 / 0.020 ft`をPASS。学習0のまま
`submission.csv`を生成し、14,151行、sample ID順、重複・欠損・finite、blend一致、SHAを検証した。
外部competition submitは行っていない。科学gate FAILとselected train anchor exp413は変更しない。
