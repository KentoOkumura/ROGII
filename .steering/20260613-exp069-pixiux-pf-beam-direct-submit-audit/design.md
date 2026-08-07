# 設計

## アプローチ

exp063 の `public_notebook_replay_audit.py` から公開 Pixiux replay feature generation を再利用する。`build_replay_test_frame()` で test rows の PF/Beam / likelihood-PF tracker features を作り、LightGBM saved booster prediction は呼ばない。`likpf_mean` を sample submission の `tvt` に map し、欠損 id は既存 fallback で埋める。

監査用に `likpf_scale_3/5/8/12`、`pf_ancc`、`pf_z`、`beam_cons`、`beam_mean`、`beam_med`、`hyb` も候補予測として保存するが、提出対象は config の `inference.candidate` だけに限定する。

## 実験範囲

- 対象実験: `exp069_pixiux_pf_beam_direct_submit_audit`
- Route: `pf_beam`
- 親実験: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 変更する変数: LightGBM `lgb_mean` ではなく `likpf_mean` raw TVT 予測を提出する。
- 固定する変数: raw competition input、Pixiux likelihood-PF replay settings、PF seeds / particles、sample submission mapping、no override / no final blend。

## リスク

- リークリスク: test-side feature generation は `TVT_input` の既知 prefix と typewell / horizontal well inputs のみを使う。train target は使わない。
- CV/LB 不一致リスク: 直接提出候補なので train-side CV は作らない。Public LB は exp063 8.811、exp027 8.781 との差分として解釈する。
- ランタイム/メモリリスク: exp063 inference と同じ feature generation を使うため hidden test rows が増えた場合は PF/likelihood-PF 計算時間が支配的になる。LightGBM inference は省くため exp063 inference よりは軽い想定。
