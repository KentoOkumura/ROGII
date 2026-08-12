# exp434 要件

## 依頼

`experiments/exp264_exp263_candidate_confidence_dual_selector/physical_model_summary.md`
で固定済みの、6 primitive、5つの50:50 pair、固定3-way blendについて、
OOFだけでなくPublic LBも同じ表で比較できるようにする。

2026-07-29の依頼では、バックログ、steering、design-only実験scaffoldを作成し、
設計を確定する。実装、Notebook採用、Kaggle package / push / run、
competition submissionはまだ行わない。

同日後続依頼`exp434を実装してください`と対象名確認により、compact
self-contained inference候補と専用testの実装まで承認された。正規Notebook採用、
Kaggle package / push / run、competition submissionは引き続き未承認とする。

さらに同日、候補ごとに同じ正規Notebookの別versionを使う方式への確認後、
ユーザーの`それでいいです。実行してください。`により正規Notebook採用、
Kaggle package / push / run、output取得、submit-checkまで承認された。
competition submissionは引き続き未承認とする。通常9 versionを完了後、
事前登録済みLikPF同一性gate failure policyが発動したため、同じLikPFを
条件付きversion 10として追加生成した。

## 目的

- exp263 Stage 1のraw-test再生成可能な12 surfaceを一切変更せず、Public LBの
  censusを完成させる。
- 既提出で候補同一性を確認できるK16、LikPF、固定3-way blendは重複提出しない。
- 未提出の5 pairと4 primitiveを、LB結果を見て候補や式を変更しない2 batchで
  提出できる設計にする。
- OOF順位とPublic LB順位の一致・逆転を記録するが、LBを使ったweight tuningや
  次候補生成は行わない。

## 制約

- Route: `pf_beam`
- 親実験:
  `exp263_last_anchor_better_candidate_confidence_pair_cache`
- 説明元:
  `experiments/exp264_exp263_candidate_confidence_dual_selector/physical_model_summary.md`
- 候補は次の12本だけとする。
  - primitive:
    `exp226_k16`、`selfgr_hmm_a070`、`likpf_mean`、`exact_hmm`、
    `pf_ancc`、`beam_mean`
  - 50:50 pair:
    `exp226_k16__selfgr_hmm_a070`、`exp226_k16__exact_hmm`、
    `exp226_k16__likpf_mean`、`selfgr_hmm_a070__likpf_mean`、
    `likpf_mean__exact_hmm`
  - fixed:
    `exp226_w500_50_50`
- OOFは3,783,989 evaluation rows / 773 wellsの保存値を使い、再計算、
  再fit、candidate追加をしない。
- pairはfloat32で`0.5 * left + 0.5 * right`、fixedは
  `0.5 * exp226_k16 + 0.25 * likpf_mean + 0.25 * exact_hmm`とする。
- hidden testではexp263 Stage 1と同じraw-test generatorを実行する。
  exposed current-test predictionや静的submissionをhidden predictionとして使わない。
- model、LightGBM config、trained fold、booster、GPU、parent/control再学習は0。
- PF/Beam/likelihood-PFの乱数はexp073由来のstable per-well seed契約を変更しない。
- 既存LBは候補同一性gateを通った場合だけ再利用する。gate不合格時は値を流用せず、
  同じ候補を追加batchで提出する。
- 1日5提出の記録済み上限を超えない。通常計画は5 pairと4 primitiveの`5 + 4`。
- batch 1のLBを見てもbatch 2の候補、順序、式を変えない。
- 実装・提出は別承認を必須とする。

## 受け入れ基準

- 12候補のID、kind、formula、OOF RMSE、OOF出典がmanifestで固定されている。
- 既存3候補はsubmission ref、kernel version、submission SHAを記録し、
  exp434再生成値との候補同一性gate結果を保存する。
- 新規提出候補は全件でrow / ID order、finite、sample schema、formula parity、
  primitive source SHA、prediction SHA、submission SHA、fallback rowsを記録する。
- 各提出は`kaggle-submit-check`をPASSしてから、別承認後にのみsubmitする。
- 12候補すべてに`existing_equivalent`または`new_submission_complete`の状態と
  Public LBが入った時点でcensus完了とする。
- Public LBは表示精度のまま記録し、同値はtieとして扱う。
- OOF順位、LB順位、rank差、`Public LB - OOF RMSE`、候補kind別の要約を
  diagnosticとして記録する。
- 実行後は`physical_model_summary.md`、`result.md`、`metrics.json`、
  `experiment_summary.md`、`SUBMISSIONS.md`を同期する。
- Public LBだけを根拠にweight、blend、selector、candidateを追加せず、
  train-side採用や最終提出候補への昇格は別判断とする。
- deterministic anchorはkernel version、input/source SHA、prediction SHA、
  submission SHA、stable seed契約がそろった候補ごとにだけ主張する。
