# exp452_scale5_likpf_direct_public_lb_audit 結果

## 状態

Kaggle private CPU inference version 1完了、取得後submit-check PASS。
Codexはcompetition submissionを行っていない。その後ユーザーが外部で提出し、
ref `55149125`がexp452であることを確認した。Kaggle statusは`COMPLETE`、
Public LBは`8.797`。

## 仮説

固定temperature-5 seed aggregationのpooled OOF改善が、train-sideで検出した
well-tailリスクよりPublic LBへ強く現れる可能性を記述評価する。

## 設定

- 親: `exp417_scale5_seed_aggregation_promotion_audit`
- 候補: `likpf_scale_5_x1p0`
- 検証: frozen train-side evidenceからdirect Public LBへのcensus
- メトリック: RMSE
- シード: 42、stable SHA256 per-well × seed index
- 学習 / model / booster: `0 / 0 / 0`

## 既存train-side証拠

| メトリック | 値 |
| --- | ---: |
| arithmetic LikPF RMSE | 11.594897884 |
| scale-5 LikPF RMSE | 10.914522073 |
| 改善 | 0.680375810 |
| 改善fold | 5/5 |
| by-well delta p95 | +2.941688483 |
| worst-well delta | +25.311274575 |
| 公開参照function parity max abs | 0.0 |
| Public LB | 8.797 |

## Kaggle inference version 1

| 項目 | 値 |
| --- | --- |
| kernel | `kentookumura/exp452-scale5-likpf-public-audit-inference` |
| version / id_no | `1 / 129271895` |
| runtime | `58.784 sec` |
| runtime設定 | private / CPU / internet off |
| rows / wells | `14,151 / 3` |
| particles / seeds | `500 / 128` |
| well-seed runs / particle trajectories | `384 / 192,000` |
| fallback rows / wells | `0 / 0` |
| public reference float32 max abs | `0.0 ft` |
| submit-check | `PASS`（FAIL 0 / WARN 0） |
| competition submission | user-submitted ref `55149125`、Codex submit 0 |

## Public LB audit

| 項目 | 値 |
| --- | ---: |
| scale-5 LikPF candidate | 8.797 |
| SHA256-seed arithmetic LikPF control（exp434 v10） | 9.807 |
| candidate改善（lower is better） | 1.010 |
| OOF改善 | 0.680375810 |
| OOF/LB方向一致 | yes |
| exp413 ML最終予測 | 7.201 |
| candidate差 vs exp413 ML | +1.596 |

- submission ref: `55149125`
- submitted at: `2026-08-01T00:00:36.783Z`
- submitted by: `kentookumura`
- attribution: ユーザー確認済み
- scoring elapsed: 完了後に監視を開始したため不明

## 実装検証

- exp413 v4が使ったexp073 PF sourceの核をself-contained sourceへ抽出し、
  source SHAとAST parityを固定した。
- 公開3 wells・14,151 rowsを500 particles × 128 seedsで再生成する専用function
  testは、exp413 v4 `likpf_scale_5`とfloat32最大差`0.0 ft`だった。
- candidate logical content SHAは
  `b713ade7adb5b185dacc941edf19aec324bcd7e075a8e903d33a23f59eb809f3`。
- 生成対象はtemperature-5の1列だけで、arithmetic mean、他temperature、ML、
  blend、selector、gate、postprocess、fallbackを含まない。
- Jupytext test、py_compile、Ruff F821、専用test、strict experiment validationはPASS。
- Kaggle生成`submission.csv`はsampleとheader、14,151行、ID順序が完全一致し、
  duplicate / missing / nonfiniteはすべて0だった。
- `submission.csv`とprediction artifactのID順序は一致し、TVT最大差は`0.0 ft`。
- outputのconfig/source、input/generation manifest、prediction gzip/decompressed、
  submissionのSHAを再計算し、保存manifestと一致した。

## 再現性

- deterministic anchor: まだ主張しない
- seed policy: stable SHA256 per-well / feature family / seed index
- kernel version: `1`、id_no `129271895`
- run config SHA: `9ecd61cf980cc8d4574e0acd55e382b4de07fec520ef660445e3d3246f7b1f12`
- source SHA: `f8777d583c9de2a5706e5c5981b418c2b51bc771d554b51b4c4ffd62122b3cf7`
- prediction logical content SHA:
  `b713ade7adb5b185dacc941edf19aec324bcd7e075a8e903d33a23f59eb809f3`
- prediction gzip SHA:
  `e87ad2e8fb219e658b99701c2e0232dbb94f4e395fa67d8045f3b5e9bf376eb5`
- prediction decompressed SHA:
  `085567c7212d00c28d3859190a25572e2de3a10f0b0df893b126f485c13679b7`
- model SHA / manifest SHA: 非該当
- submission SHA:
  `4ace1476ac4777fd7fc17742f3d3786ae05f1c436e35e37ae3f4348350f51217`
- rerun result: 未実行

## 解釈

凍結scale-5 surfaceのKaggle inferenceと公開参照parityは成立した。Public LBでも、
同じSHA256 seed familyのarithmetic LikPF control `9.807`に対して`8.797`となり、
scale-5集約を`1.010`改善の方向で支持した。OOFの`0.680376 ft`改善と方向は一致する。

ただしPublic LBは小さい公開splitの記述censusであり、exp417のby-well p95
`+2.941688 ft`、worst `+25.311275 ft`というtail FAILを覆さない。自動昇格、
temperature/seed/particle/weight変更、LBを見た救済は行わない。exp413の`7.201`は
downstream ML最終予測でrouteが異なり、本候補のdirect primitive scoreとは混ぜない。

## 次

この実験では追加run、rerun、再提出、LB適応を行わない。
生成物は`/tmp/kaggle-output/exp452_scale5_likpf_direct_public_lb_audit/inference_v1`
で検証済みだが、リポジトリへ大きなoutputを常設しない。
