# 要件

## 依頼

`ravaghi_vs_pixiux_lgbm_feature_parity_audit` を実装する。Ravaghi public
LightGBM notebook と pixiux dual-pipeline blend notebook の LightGBM 入力特徴量を、
raw competition files から公開実装に近い形で再生して比較する。

追加要件として、学習後に全 fold の平均を取った特徴量重要度の可視化を保存する。

追加依頼として、exp063 内に inference notebook を実装する。hidden-compatible branch は不要とし、
train audit で選ばれた Pixiux likelihood-PF LightGBM replay candidate を public replay 条件のまま
test に適用して `submission.csv` を作る。
inference では train features を再生成せず、train notebook が保存した fold booster を読む。
後続実験の train で再利用できるよう、生成済み PF/Beam/likelihood-PF tracker features を保存する。

## 制約

- Route: `ml_model`
- 親生成物は使わず、competition raw train files から公開 notebook 由来の feature builder を実行する。
- 比較条件は同一 train wells、同一 GroupKFold split、同一 target `TVT - last_known_tvt`、public LightGBM configs、raw OOF prediction のみ。
- stack / Ridge / CatBoost / final blend / projection / static visible override / pretrained booster は使わない。
- pixiux 側は Ravaghi-style public base features に likelihood-PF delta features を加えた差分として評価する。
- train は `pixiux_likpf_public_replay` の fold LightGBM booster と PF/Beam/likelihood-PF tracker feature frame を保存する。
- inference は `pixiux_likpf_public_replay` `lgb_mean` を対象にし、saved booster と test-side features だけを使う。hidden-specific branch、
  guarded overlap override、static visible override、pretrained booster、CatBoost、Ridge stack、
  final public notebook blend、projection postprocess は含めない。

## 受け入れ基準

- `experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/` に config、settings、train/inference notebook、監査 script、記録ファイルがある。
- train notebook は `public_notebook_replay_audit.py` を呼び、目的、設定確認、raw input 確認、audit 実行、生成物確認のセル構成を持つ。
- inference notebook は目的、設定確認、raw input 確認、public replay inference 実行、submission/生成物確認のセル構成を持つ。
- 生成物として metrics、fold/model 別 feature importance、全 fold/model 平均 feature importance、matplotlib plot、OOF prediction、feature schema、summary JSON を保存する。
- train 生成物として saved LightGBM booster manifest と reusable tracker train feature frame を保存する。
- inference 生成物として submission、inference metrics、test predictions、feature schema、summary JSON、reusable tracker test feature frame を保存する。
- `py_compile`、`ruff check`、`scripts/validate_experiment.py --experiment exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit` が通る。
