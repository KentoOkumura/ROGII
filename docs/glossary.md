# 用語集

| 用語 | 意味 | メモ |
| --- | --- | --- |
| CV | 交差検証スコア | ローカル検証の結果。 |
| LB | リーダーボードスコア | Kaggle の public/private スコア。 |
| 検証リークリスク | 推論時に使えない情報が検証に混入している可能性。 | 旧表記: Leak risk。該当する実験は `leak-risk` として記録する。 |
| 利用可能 | ユーザーが派生実験や提出に使えると判断した状態。 | 旧表記: Usable。エージェントだけでは確定しない。 |
| 非推奨 | ユーザーが履歴として残すが再利用しないと判断した状態。 | 旧表記: Deprecated。エージェントだけでは確定せず、可能なら代替実験を併記する。 |
| TVT | True Vertical Thickness | このコンペの target。train では `TVT`、submission では `tvt`。 |
| TVT_input | 既知区間の target copy | evaluation zone では NaN。提出対象行の判定にも使う。 |
| MD | Measured Depth | wellbore の長さ方向の深度。単位 ft。 |
| GR | Gamma Ray | 岩石の自然放射能ログ。欠損がある。 |
| Typewell | vertical reference log | horizontal well と対応し、`TVT`, `GR`, `Geology` を持つ。 |
| well_id | well の識別子 | ファイル名の 8 文字 hash から作る。CV の group key。 |
| Evaluation zone | 提出対象区間 | `TVT_input` が NaN の区間。 |
| Formation columns | `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` | 公式説明では Training only。推論特徴として直接使わない。 |
| 基準、基準スコア、基準予測 | 比較の起点にするスコアや予測。 | 旧表記: anchor。 |
| 比較基準 | 変更案と比べる固定の条件や候補。 | 旧表記: control。 |
| 生成物 | 実験で保存した特徴量、予測、metrics、ログなど。 | 旧表記: artifact。 |
| 見えない test well 用処理 | Kaggle の本番採点で出てくる、train に存在しない test well だけに使う推論処理。public sample の 3 well は train 由来なので、この処理が動かないことがある。 | 以前は hidden branch と書いていた。実験名・ファイル名・kernel id では `hidden_branch` や `hidden-branch` が残る。 |
| 条件を揃えること | train 側と推論側で seed、粒子数、特徴列、欠損処理などを一致させること。 | 旧表記: parity。 |
| train well の途中以降を隠した疑似 test 条件 | train well の途中から先の `TVT_input` を NaN にして、本番 test のように予測させる代理条件。 | 以前は pseudo-hidden と書いていた。 |
| 代理検証 | 本番採点の代わりに、train well の途中以降を隠した疑似 test 条件で挙動を見る補助検証。 | 旧表記: surrogate audit。 |
| 評価条件 | 同じ候補を比較するためのデータ、split、行集合、後処理条件のまとまり。 | 旧表記: evaluation surface。 |
| 重み調整 | 候補予測をどの程度混ぜるかを決める処理。 | 旧表記: gate。 |
| 信頼度 | 予測や候補をどれだけ信用するかを示す特徴や診断値。 | 旧表記: confidence。 |
| 推論側へ移植 | train 側で選んだ処理を inference notebook に入れて動かすこと。 | 旧表記: port。 |
| 直接置き換え | 既存の基準予測を別候補で丸ごと置き換えること。 | 旧表記: direct replacement。 |
