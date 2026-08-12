# 要件

## 依頼

`exp092_u_projection_correction_disagreement_fullrun` の残タスクだった worst-well gating / regression guard と visible-test feature parity を、提出物を作らない監査 script として実装する。

OOF 側では `exp092_lgb1` が全体 RMSE と Public LB を改善した一方、`b8c49c1a` など一部 well で exp077 比 +4 RMSE 以上の悪化が出ている。通常の Kaggle notebook 実行で読める exposed sample / visible test には正解がないため、OOF worst-regression wells の補正量・step continuity・prediction range を基準に、exp092 inference の visible test wells が異常な補正分布や schema drift を示していないかを判定する。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 既存 `exp092_u_projection_correction_disagreement_fullrun` の follow-up として扱い、duplicate exp フォルダは作らない。
- guard は submission.csv を生成・変更しない。
- Code Competition 形式では hidden LB test は code submission rerun 時に差し替えられる。通常 kernel の guard は hidden LB test を観測できない。
- visible-test 正解は使えないため、判定は parity / distribution / continuity warning とし、CV 改善や hidden LB 安全性を主張しない。
- hidden test 側で見たい事象は、提出 notebook 内の assert probe として別途設計し、submission が通ったか落ちたかだけを観測可能な信号として扱う。
- 入力が未取得の場合でも、必要な Kaggle output path と不足理由が分かるエラーにする。

## 受け入れ基準

- exp092 inference prediction、optional exp073 / exp077 inference surface、OOF delta guard、train/inference feature schema、projection feature summary を読める visible-test guard script がある。
- visible test well ごとの prediction step、exp073/exp077 比 correction、tail bucket、prefix anchor parity、prediction range、schema parity を CSV / JSON に保存する。
- OOF worst-regression wells の補正量分布から visible-test warning thresholds を作り、test well の警告数と代表 well を summary に記録する。
- deterministic anchor ではなく監査生成物として扱う。生成物には入力 raw SHA と gzip decompressed SHA、prediction SHA を記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
