# 要件

## 依頼

ユーザーの「別案: 井戸全体のGRを使用 → bidirectional smoother」について、バックログ、独立した実験ディレクトリ、steeringを作成し、実装前の設計を確定する。科学コード、正規Notebookロジック、Kaggle package/push/run、inference、submissionはまだ作成・実行しない。

## 背景

`exp345_exp209_time_varying_gr_affine_calibration_hmm`は、現在行までのraw GRだけで`[b_t, log(a_t)]`を更新するcausal EKFをexp209 exact-HMMのGR観測中心へ適用した。last-640 Stage 0はpooled`+0.169505 ft`、4/5 folds、GR NLL、boundaryをPASSしたが、400/773 wells悪化、worst`+9.354827 ft`、hidden-like 2 scope欠落でscientific AND gateをFAILし、terminal closeした。

本案は、推論時に井戸全体のraw GRが利用できるコンペ契約を使い、exp345のforward affine stateを未来GRからも後ろ向きに平滑化すれば、pooled gainを保ちながらwell間不安定性とworst tailを抑えられるかを独立に検証する。

## 必須要件

- 実験番号は`exp350_exp345_bidirectional_gr_affine_smoother`、Routeは`pf_beam`とする。
- exp345は閉鎖のまま維持し、そのKaggle version 2成果物をimmutable controlとして参照する。exp345をreopen、reparent、再pushしない。
- 単一の科学変更は、exp345 causal EKF後のaffine stateに固定区間extended RTS backward passを1回追加することだけとする。
- state、初期fit、process noise、観測式、exp209 HMM grammar、sigma、missing weight、transition、prior、posterior-mean decoderをexp345から変更しない。
- Stage 0ではlast-640の`TVT_input`だけをmaskし、raw GRはscore suffixを含む井戸全体で保持する。future raw GRは許可するが、future/true TVT、error、formation、fold、hidden-like roleはprediction freeze前に使用しない。
- exp345保存parent mean/std、process-noise table、causal schedule、parent/causal predictionをSHAで固定する。
- exp345 forward scheduleを再生成し、保存済みcausal scheduleと`1e-10`以内で一致した場合だけsmootherへ進む。
- smoother scheduleと新candidate predictionをtruth-freeでfreezeし、SHA取得後にだけtruth、fold、hidden-like roleをjoinする。
- Stage 0の新規計算は1 scientific variant、773 forward filter、773 smoother、773 candidate HMM runs。parent/causal control HMM再実行、LightGBM config、学習fold、booster、PF、Beam、GPUはすべて0とする。
- GR reconstruction NLLはcurrent/future GRを同じsmootherが利用するため、promotion metricではなく診断値だけにする。
- Stage 0全gate PASS後もStage 1、inference、submissionへ自動移行せず、別承認を必須とする。

## Promotion gate

### Technical

- exp345入力artifactの全事前SHA一致。
- forward scheduleのscale/intercept parity最大絶対差`<=1e-10`、raw-GR update mask完全一致。
- 保存済みparent/causal metric parity絶対差`<=1e-10`。
- prediction 494,720 rows / 773 wells、全finite、new HMM 773/773。
- smoother terminal stateとforward terminal stateの差`<=1e-12`。
- smoothed covarianceの最小固有値`>=-1e-8`、`P_smooth-P_filter`の最大正固有値`<=1e-8`。
- smoothed scaleの`[0.25,4.0]` clip率`<=1%`。
- Kaggle CPU runtime`<=8.5 h`。

### Scientific

- masked exp209 parent比pooled改善`>=0.05 ft`。
- exp345 causal比pooled改善`>=0.02 ft`。
- masked parent比・causal比とも改善fold`>=4/5`。
- hidden-like spatial / typewell-purgedの2 scopeを必ず生成し、両baselineに対して各scope非悪化。
- parent比by-well deltaのmedian`<=0 ft`、p95`<=0 ft`。
- parent比worst-well delta`<=+0.25 ft`。
- visible-prefix境界でのaffine observation jump p95`<=3 sigma`。

全項目のAND gateとし、FAIL時は`stage_0_failed_close_without_rescue`で閉じる。

## 禁止事項

- process noise、initial fit、slope bound、RTS式、pseudoinverse tolerance、covariance floorのgrid。
- causal/bidirectional prediction blend、row/well gate、fallback selector。
- finite-only/MAD sigma、missing-GR downweight、well別`sig_r`、Type Well群prior、joint TVT/rate/affine state。
- true TVT/error/oracleを使うstate、smoother、clip、gate、停止条件。
- forward-backward反復、HMM結果をbase pathへ戻す再fit。
- Stage 0 FAIL後のparameter rescue、version追加、inference、submission。

## Design-only受け入れ基準

- `config.yaml`、`requirements.md`、`design.md`、`tasklist.md`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`に同じ設計・実行量・禁止事項が記録されている。
- `KAGGLE_DIRECTION.md`の未着手バックログへ追加され、既存P1/P2を追い越さない低・P3として位置付けられている。
- `experiment_summary.md`へ`design_frozen_not_implemented`として反映されている。
- strict experiment validationとexperiment docs reviewerを通す。
- 実装ファイル、Kaggle package、run、prediction、submissionを生成していない。

## 2026-07-23 実行結果

後続のユーザー承認によりdesign-only境界を解除し、固定済みStage 0だけを実装・実行した。Kaggle version 1はtechnical PASS / scientific FAILで、decisionは`stage_0_failed_close_without_rescue`となった。Stage 1、inference、submission、post-hoc救済は引き続き禁止する。詳細は実験`result.md`と`SESSION_NOTES.md`を正とする。
