# 要件

## 依頼

`exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit` の strict Pixiux replay inference v2 を提出候補として扱えるか、CV・推論完了・submit-check・train/test overlap probe の根拠を 1 つのゲートで判定する。

ユーザー指定は `exp063_cv_submit_gate` だが、既に `exp065` まで存在するため、新規実験名は `exp066_cv_submit_gate` とする。実験名には `exp063` を付けない。

## 制約

- Route: `ml_model`
- 新しい学習や予測生成はしない。提出候補は `exp063` inference v2 の保存済み booster inference output とする。
- `submission.csv` を `exp066` 配下へ常設コピーしない。判定結果と提出対象 kernel/version/path の参照だけを保存する。
- `exp064_train_test_well_id_assert_probe` の hidden scoring assertion が発火しなかったことを、static visible override を使わない提出候補の補助根拠として扱う。
- Public LB は未確認なので、ゲート通過後も LB anchor は更新しない。

## 受け入れ基準

- `config.yaml` に gate しきい値、親実験、probe 実験、提出対象 kernel/version を明記する。
- gate 実行で `artifacts/cv_submit_gate_decision.json`、`artifacts/cv_submit_gate_decision.csv`、`artifacts/cv_submit_gate_report.md` を保存する。
- 判定条件には少なくとも CV 上限、Ravaghi replay への改善幅、推論完了、submit-check PASS、fallback 0、行数一致、予測範囲 sanity、exp064 no-overlap probe 完了を含める。
- gate が通った場合でも、実提出は自動実行せず、提出コマンド候補を report に記録する。
