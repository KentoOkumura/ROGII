# exp434_physics_candidate_public_lb_audit

## 状態

- ルート: `pf_beam`
- 状態: 正規inference採用済み・Kaggle version 1–10完了・10/10採点完了
- 優先度: P1、late-stage LB census
- OOF: 既存12候補の保存値を使用
- Public LB: 全12候補確定
- Private LB: -
- 作成日: 2026-07-29
- 親実験: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- steering:
  `docs/legacy/steering/20260729-exp434-physics-candidate-public-lb-audit/`

## 仮説

exp263でhidden-safeに再生成できる6 primitive、5つの固定50:50 pair、
固定3-way blendは、OOF上で異なる精度と補完性を示した。候補と式を結果前に
固定してPublic LBを測れば、OOF順位がLBでも維持されるかを、weight tuningと
切り離して確認できる。

## 変更点

- 新しい物理候補、OOF、modelは作らない。
- 12候補を同一のexp263 Stage 1 generator契約でLB監査する。
- 既存同一候補のK16、LikPF、固定3-wayは同一性確認後に再利用する。
- 未提出の5 pairと4 primitiveを凍結順序で提出し、日次枠に合わせて実際は
  3回に分けた。同一性gate不合格のLikPFも事前登録済みpolicyどおり追加提出した。
- Jupytext percent形式の
  `exp434_physics_candidate_public_lb_audit_compact_selfcontained_inference.py`
  と候補Notebookを実装した。
- 6 generator source SHA、exp226 source config SHA、exp263 Stage 0 / Stage 1
  provenance、exposed reference gzip content SHAをfail-fastで照合する。
- 12候補manifest、float32 formula parity、既存3提出との同一性gate、
  1-version 1-candidate選択、prediction/submission/candidate-version SHAを実装した。
- `selected_candidate`は通常9候補だけを許可し、K16 / LikPFは同一性gate failureを
  configへ明示した場合だけ追加候補として許可する。固定3-wayの再提出は拒否する。

## 検証方針

- OOF:
  3,783,989 rows / 773 wellsの保存済みGroupKFold評価
- LB:
  hidden-safe Kaggle code submissionのPublic LB
- technical:
  source SHA、row/ID、finite、formula parity、fallback、prediction/submission SHA
- diagnostic:
  `LB - OOF`、OOF/LB rank差、kind別要約、Spearman順位相関
- leakage:
  hidden testでraw inputから再生成し、静的exposed predictionを使わない

## 現在わかっているLB

| 候補 | OOF RMSE | Public LB | 出典 |
| --- | ---: | ---: | --- |
| `exp226_k16` | 9.427110 | 9.837 | exp226 ref `54491603`、最大差`0.000488265 ft`でgate PASS |
| `exp226_k16__selfgr_hmm_a070` | 8.532715 | 7.913 | exp434 ref `55083262` |
| `exp226_k16__exact_hmm` | 8.635074 | 7.678 | exp434 ref `55083266` |
| `exp226_k16__likpf_mean` | 8.813822 | 8.365 | exp434 ref `55083270` |
| `selfgr_hmm_a070__likpf_mean` | 10.123457 | 8.812 | exp434 ref `55105249` |
| `likpf_mean__exact_hmm` | 10.269697 | 8.642 | exp434 ref `55105256` |
| `selfgr_hmm_a070` | 11.349943 | 9.318 | exp434 ref `55105261` |
| `likpf_mean` | 11.594898 | 9.807 | exp434 ref `55133074` COMPLETE。SHA256 seed版。exp069 v3 ref `53706005`とは最大`4.77832 ft`不一致、流用不可 |
| `exact_hmm` | 11.938287 | 9.063 | exp434 ref `55105266` |
| `pf_ancc` | 14.493051 | 12.061 | exp434 ref `55133068` COMPLETE |
| `beam_mean` | 15.774327 | 15.563 | exp434 ref `55133072` COMPLETE |
| `exp226_w500_50_50` | 8.238331 | 7.800 | exp263 ref `54761954`、exact parity済み |

10候補は候補別Kaggle versionの生成とsubmit-checkまで完了した。
凍結順序のversion 1–10を提出し、全件の採点が完了した。

## 実行入口

正規inference Notebookへcompact self-contained実装を採用済みである。

- Jupytext source:
  `exp434_physics_candidate_public_lb_audit_compact_selfcontained_inference.py`
- 正規Notebook:
  `exp434_physics_candidate_public_lb_audit_inference.ipynb`
- Kaggle kernel:
  `kentookumura/exp434-physics-candidate-lb-audit-infer`
- 実行台帳:
  `kaggle_run_ledger.json`

通常9候補とLikPF同一性FAILによる条件付き1候補の計10 versionを実行し、
10/10でKaggle `COMPLETE`、output取得、submit-check PASSを確認した。
competition submissionはversion 1–10まで承認・実施し、全10件の
`COMPLETE`とPublic LBを確認した。

## 所見

### 良かった点

- OOFで比較した12候補とLB対象を完全に一致させた。
- 既提出3候補を無条件に再提出せず、同一性gateを設けた。
- 5 pair / 4 primitiveの順序をLB観測前に固定した。
- 同一exp helper importに依存せず、入力確認から生成物manifestまで11章で追える
  self-contained候補を実装した。
- 専用contract test 8件、Jupytext test、py_compile、Ruff F821を通した。
- 10 versionすべてで14,151 rows / 3 wells / fallback 0、親式parity
  `0.0 ft`、candidate bank SHA一致を確認した。

### 悪かった点

- 実際に通常9件とLikPF追加1件の計10提出枠を使った。
- Public LBは候補間相関とsplit差を含み、773-well OOFの代替ではない。

### リスク / 注意

- Batch 1のLBを見てもBatch 2や式を変更しない。
- LB上位候補から新しいblendやweightを作らない。
- train-side採用や最終提出への昇格はこの実験の目的外。

## 次

全12候補のPublic LB censusは完了した。最良はK16 + exact HMMの`7.678`、
OOF/LB Spearman順位相関は`0.846154`である。LBからweight tuningや候補の
自動昇格は行わず、後続候補の記述的な比較基準として使う。
