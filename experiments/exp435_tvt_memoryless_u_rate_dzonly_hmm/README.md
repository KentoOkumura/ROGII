# exp435 TVT memoryless U-rate / dz-only HMM

## 状態

- Route: `pf_beam`
- 状態: Stage 0 technical PASS / 全variant mechanism FAIL、branch閉鎖
- CV / LB: なし
- 作成日: 2026-07-29
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

exp209のpersistent TVT offsetは、rate posteriorの追従遅れを含む
forward transition / prior hysteresisが主因だった。TVT確率分布だけを持続状態とし、
41個のU-rate候補を毎行独立に周辺化すれば、非ゼロrateの表現力を残しながら
rate履歴を除ける可能性がある。

さらに同じposition-only kernelを`r_U=0`へ固定したdz-onlyと比較し、
非ゼロU-rate support自体が必要かを切り分ける。

## 比較

| モデル | 次行へ伝える状態 | U-rate | rate履歴 |
| --- | --- | --- | --- |
| 保存済みexp209 | `(TVT, r_U)` | 41 states | あり |
| `memoryless_41rate` | TVT確率分布 | 41候補を毎行周辺化 | なし |
| `dz_only_r0` | TVT確率分布 | `r_U=0` | なし |

`memoryless_41rate`と`dz_only_r0`の両方を同じ実験、同じfixed32で実行する。
TVT posterior mean 1点だけを再帰する設計にはしない。

## 固定設計

- `U=TVT+Z`
- `ΔTVT=r_U ΔMD-ΔZ`
- TVT grid、GR emission、position kernel、initial TVT prior、
  forward-backward、posterior readoutはexp209から固定
- memoryless rate gridは親と同じ41点
- rate重みは`mom=0.998`, `sig_r=0.002`から導くzero-centered stationary分布
- dz-onlyは同じkernelの`rates=[0]`, `weights=[1]`特殊ケース
- prefix `init_rate`はsupport幅にだけ使用し、rate重みの中心には使わない
- 新しいabsolute TVT anchorは追加しない

## 検証方針

Stage 0はexp411 fixed32
（persistent 16 / matched control 16）でmechanismだけを確認する。

- treatment: 2
- HMM runs: `2 × 32 = 64`
- parent rerun: 0
- model / booster / PF / Beam / GPU: 0

Stage 0を通過したvariantだけ、別承認後にfull 773-well OOFへ進める。
fullではdirect exp209比較に加え、exp263固定式のHMM成分だけを置換して評価する。
weight tuningは行わない。

fixed32は誤差情報で選ばれたsampleなのでCVとは呼ばない。suffix truth、role、
fold、episode、cause、errorはcandidate prediction freeze後にだけjoinする。
hidden-like roleはStage 0では読み込まず、別承認後のfull OOFでだけ評価する。

## 実装

- compact self-contained Jupytext train / inference guardを実装済み
- 正規train / inference Notebookへ採用済み
- TVT-only forward-backward、stationary 41-rate mixture、dz-only特殊ケースを
  同一position kernelで実装
- exp408 cause付きepisode ledgerとexp411 fixed32をSHA固定
- role / fold / truth / episode / causeは両variantのprediction / diagnostic
  freeze後にだけ読み込む
- 専用contract test 11件、構文、Ruff F821、Jupytext round-trip、
  strict experiment validationを通過

edge-rate生成物は各行で使う固定priorのmean / std / edge massであり、
filtered / smoothed rate posteriorではない。rate responsibilityを次行へ保持しない
ことをstate shapeとcontract testで監査する。

## 所見

- Kaggle private CPU version 1（id_no `129049294`）を完了した。
- technical gateは全PASS。runtime `46.077 sec`、peak RSS `0.455 GB`。
- memoryless / dz-onlyはいずれもforward-causeとpersistent episodeのpooled SSEを
  改善したが、改善wellは`4/16` / `5/16`、改善foldは双方`1/5`。
- matched-control pooled RMSE deltaは`+16.151527` / `+13.705216 ft`、
  by-well p95は`+29.129905` / `+24.955652 ft`で大幅にFAILした。
- exp424の結果から、rate mean reversionを弱めるだけでは不十分だった。
- exp355の結果から、非ゼロrateには平均signalがあり、dz-only単独へ直行せず
  memorylessとの対比較が必要である。
- dz-onlyは新しいabsolute TVT anchorを追加しないため、reanchorとは扱わない。
- 両variantのStage 1 eligibilityはfalse。same-OOF救済なしでbranchを閉じる。

## 実行入口

- train notebook:
  `exp435_tvt_memoryless_u_rate_dzonly_hmm_train.ipynb`
- inference notebook:
  `exp435_tvt_memoryless_u_rate_dzonly_hmm_inference.ipynb`

Notebookは実装・実行済み。Stage 1、inference、submissionは無効。

## 次

両variantをStage 0 FAILとして閉じ、Stage 1、inference、submissionへ進めない。
同一fixed32でrate weight / support / noise / gateを救済しない。
