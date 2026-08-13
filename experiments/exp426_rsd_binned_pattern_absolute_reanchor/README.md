# exp426_rsd_binned_pattern_absolute_reanchor

## 状態

- ルート: `pf_beam`
- 状態: `completed_stage_a_technical_failed_closed`
- 優先度: P3
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-28
- 親実験:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- PF親:
  `exp404_scale5_sigma_gr_likelihood_pf_ablation`

## 仮説

論文のRSD 0.5 ftビニング＋GR pattern correlationは、pointwise emissionの置換よりも、
local rate modelが失ったcoarse absolute datumを再観測する用途に適する。
そのscoreがoffsetを識別できれば、exp226の累積offsetを再アンカーし、PFでは
複数datum basinへparticle supportを安全に追加できる可能性がある。

## 変更点

- Stage A:
  exp226 final pathへ固定13 offsetsを加え、RSD bin mean Pearsonで順位付けする。
- Stage B:
  同じscoreをcoarse datum Viterbiへ入れ、exp226 local pathへ連続補間した
  absolute correctionだけを加える。
- Stage C:
  同じtop-3 datum basinを、元continuation 90%・全13 anchor support付きの
  importance-corrected likelihood-PF proposalへ使う。
- exact HMM emission、exp226 local rate / shape、PF raw GR emissionは変更しない。

## 検証方針

- Fold: exp226と同じ5 folds
- Group: `well_id`
- Stage A/B: 全773 OOF wells、truth-late、0 model / 0 PF
- Stage C0: exp410 fixed 12 sentinel wells、uniform / guided paired PF
- Stage C1: C0 PASS後だけ全773 wells / 4 CPU shards
- Leakage check:
  score / rank / top-3 / prediction / SHAをfreezeするまでtruth、error、
  oracle offset、episode、hidden-like roleを読まない。

## 実行量

| Stage | 新規実行 |
| --- | --- |
| A | score 1、descriptive 2、control 3、model/PF/HMM 0 |
| B | deterministic prediction 1、model/PF/HMM 0 |
| C0 | PF 2 variants × 12 wells = 24 well-runs |
| C1 | PF 1 variant × 773 wells、500 particles ×128 seeds |

保存済みexp226 / exp404 / exp209 controlは再実行しない。

## 実行入口

- Stage A compact self-contained候補:
  `exp426_rsd_binned_pattern_absolute_reanchor_compact_selfcontained_train.py`
  / `.ipynb`
- 正規学習 notebook:
  `exp426_rsd_binned_pattern_absolute_reanchor_train.ipynb`
- 推論 notebook:
  `exp426_rsd_binned_pattern_absolute_reanchor_inference.ipynb`
- Stage A実装とcontract testsは完了し、compact self-contained候補を
  正規train Notebookへ採用した。
- Kaggle private CPU version 1を実行し、technical gate FAILでfail-closeした。
- Kaggle Notebook実行を正とし、ローカル実行は明示依頼されたsmoke debugだけにする。

## 結果

| メトリック | 値 |
| --- | --- |
| Stage A | technical FAIL・完了 |
| supported blocks | `25.593939%`（gate `>=95%`） |
| supported wells | `89.262613%`（gate `>=98%`） |
| runtime / peak RSS | `164.719113 sec` / `0.803265 GB` |
| Stage B | 未実装・停止 |
| Stage C0 / C1 | 未実装・停止 |
| Public / Private LB | - / - |

## 所見

### 判定

- 3,783,989 rows / 773 wells / 7,787 blocksのtarget-free score生成を完了した。
- inventory、順序、重複、finite score、rank / top-3、runtime、memory、
  fixed-probe parityはPASSしたが、support 2条件がFAILした。
- fail-closeによりtruthとhidden-like roleはfreeze前後とも未読で、
  scientific評価は実施していない。
- これはoffset識別精度ではなく、識別性評価の前提となる観測coverageの
  technical FAILである。

### 利用可否

- prediction、feature、HMM residual-datum state、PF candidateには利用しない。
- Stage B / C、inference、submissionには進まない。

## 注意

- exp280 raw shift matchingとexp360 ZNCCはnegativeであり、RSD binningの成功を
  前提にしない。
- Stage AはHMM residual-datum stateの共通必要条件だが、exact HMM改善の証明ではない。
- 512-row全体を使うbatch scoreであり、real-time causal geosteeringとは呼ばない。
- 前段FAIL時はparameter rescueせず、後段を実装しない。

## 次

exp426をterminal closeする。同じOOFでbin幅、block、offset、support条件、
Type Well extrapolation、score familyを変更する救済や、同じRSD-binned
score familyの後続backlogは追加しない。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、
実験名や設定名を除いて日本語優先で記録する。
