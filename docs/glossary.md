# 用語集

| 用語 | 意味 | メモ |
| --- | --- | --- |
| CV | 交差検証スコア | ローカル検証の結果。 |
| LB | リーダーボードスコア | Kaggle の public/private スコア。 |
| EDA | Exploratory Data Analysis | データ分布、欠損、誤差などを探索的に確認する分析。 |
| ML | Machine Learning | データから予測規則を学習する手法。 |
| RMSE | Root Mean Squared Error | 二乗誤差平均の平方根。単位は予測対象と同じ。 |
| AUC | Area Under the ROC Curve | 二値の順位判別を測るROC曲線下面積。対象labelを併記する。 |
| NLL | Negative Log-Likelihood | 観測値に対する負の対数尤度。小さいほど観測への尤度が高い。 |
| SSE | Sum of Squared Errors | 二乗誤差の合計。比較する行数と重みを併記する。 |
| IAT | Integrated Autocorrelation Time | 系列の自己相関を等価な相関長として要約した値。算出方法を併記する。 |
| p95 | 95th percentile | 分布の95パーセンタイル。対象単位がrowかwellかを併記する。 |
| OOF | Out-of-Fold prediction | 各train行を、その行が属するfoldを学習に使っていないモデルで予測した値。 |
| PF | Particle Filter | 粒子集合で状態分布を近似する逐次モンテカルロ法。 |
| HMM | Hidden Markov Model | 観測されない状態と観測値の確率的な関係を表すモデル。 |
| DTW | Dynamic Time Warping | 二つの系列を非線形に対応付ける動的時間伸縮法。 |
| RNG | Random Number Generator | 乱数生成器。再現性を管理するときはseedと生成単位を記録する。 |
| SHA | Secure Hash Algorithmによるhash値 | 内容同一性の証拠。使用したalgorithmと、圧縮前後のどちらをhashしたかを明示する。 |
| KKB | Kaggle Notebookの実行基盤 | このリポジトリ内でKaggle Kernel Backendを指す略記。ユーザーへの初出ではKaggle Notebook実行と併記する。 |
| `検討メモ・設計不可` | 結果や実装方針に影響する未決事項が残るbacklog状態。 | このリポジトリ内の管理用語。推測で実装せず、ユーザー確認を行う。 |
| `設計可能・実験化未承認` | 実験境界が揃い未決事項がないが、実験化は承認されていないbacklog状態。 | このリポジトリ内の管理用語。 |
| `planned` / `running` / `debug_completed` / `scaffold_completed` / `failed` | 実験の実行状態。 | `metrics.json`のstatusに記録する。状態変更の規則は`AGENTS.md`を正とする。 |
| `usable` | ユーザーが派生実験や提出に使えると判断した実験status。 | エージェントだけでは確定しない。 |
| `completed` | ユーザーが必要な検証と記録を終えたと判断した実験status。 | エージェントだけでは確定しない。 |
| `deprecated` | ユーザーが履歴として残すが再利用しないと判断した実験status。 | エージェントだけでは確定せず、可能なら代替実験を併記する。 |
| `discarded` | ユーザーが候補として使わないと判断した実験status。 | エージェントだけでは確定しない。 |
| `leak-risk` | 推論時に使えない情報が検証に混入している可能性を示す実験status。 | 採用・不採用・完了の判断ではない。 |
| `faithful` | 参照手法の手法契約に含まれる本質的な要素を、対象範囲ですべて実装する実装区分。 | このリポジトリ内の管理用語。実際の処理と対象範囲を先に説明する。 |
| `staged-faithful` | target、output表現、loss、decode、context unitを保ち、データ量、fold数、epoch数、解像度などだけを縮小する実装区分。 | このリポジトリ内の管理用語。縮小した項目を列挙する。 |
| `proxy` | target、output表現、loss、decode、whole-group / local contextのいずれかを変更または省略する実装区分。 | このリポジトリ内の管理用語。検証できない主張とユーザー承認を記録する。 |
| `parameter` | 処理構造を保ち、設定値だけを変更する変更class。 | このリポジトリ内の管理用語。 |
| `add-only` | 親実験の入力や特徴を残したまま新しい入力や特徴を追加する変更class。 | このリポジトリ内の管理用語。モデル等も変える場合は変更内容を別途明記する。 |
| `selector-only` | 固定した候補生成を変えず、候補の選択処理だけを変更する変更class。 | このリポジトリ内の管理用語。 |
| `postprocess` | 基礎予測の生成後に適用する補正だけを変更する変更class。 | このリポジトリ内の管理用語。 |
| `mechanism` | モデル構造、損失、状態遷移、候補生成など予測機構を変更する変更class。 | このリポジトリ内の管理用語。 |
| `representation` | input、targetまたはoutputの表現を変更する変更class。 | このリポジトリ内の管理用語。 |
| TVT | True Vertical Thickness | このコンペの target。train では `TVT`、submission では `tvt`。 |
| TVT_input | 既知区間の target copy | evaluation zone では NaN。提出対象行の判定にも使う。 |
| MD | Measured Depth | wellbore の長さ方向の深度。単位 ft。 |
| GR | Gamma Ray | 岩石の自然放射能ログ。欠損がある。 |
| Typewell | vertical reference log | horizontal well と対応し、`TVT`, `GR`, `Geology` を持つ。 |
| well_id | well の識別子 | ファイル名の 8 文字 hash から作る。CV の group key。 |
| Evaluation zone | 提出対象区間 | `TVT_input` が NaN の区間。 |
| Formation columns | `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` | 公式説明では Training only。推論特徴として直接使わない。 |
| anchor | 比較の起点にするスコアや予測。 | このリポジトリ内の比較用語。routeごとに根拠を併記する。 |
| `ml_model` route | 学習済みモデルが最終予測を生成する実験分類。 | このリポジトリ内の管理用語。PF、Beam、HMM由来の値を特徴量として使っても、最終予測が学習済みモデルの出力ならこの分類にする。 |
| `pf_beam` route | PF、Beam、HMMなどの系列推定が最終予測を直接生成する実験分類。 | このリポジトリ内の管理用語。系列推定の出力を後段の学習済みモデルへ入力する場合は含めない。 |
| `ensemble` route | 複数の完成した予測をblendまたはstackして最終予測を生成する実験分類。 | このリポジトリ内の管理用語。公開Notebookの再現は一律にこの分類にせず、再現した最終予測の生成方法で分類する。 |
| control | 変更案と比べる固定条件または候補。 | 実験設計で一般に使われる対照条件。 |
| artifact | 実験で保存した特徴量、予測、metrics、modelなどの出力。 | 配置規則は`AGENTS.md`を正とする。 |
| hidden test | Kaggleの本番採点時にだけ与えられるtest data。 | public sampleとデータ構成が異なる場合がある。 |
| parity | trainとinferenceでseed、粒子数、特徴列、欠損処理などの条件を一致させること。 | 何を一致させたかを具体的に記録する。 |
| train well の途中以降を隠した疑似 test 条件 | train well の途中から先の `TVT_input` を NaN にして、本番 test のように予測させる代理条件。 | 以前は pseudo-hidden と書いていた。 |
| surrogate validation | 本番採点の代わりに、train wellの途中以降を隠した疑似test条件で挙動を見る補助検証。 | 本番評価ではないことを明記する。 |
| evaluation setting | 同じ候補を比較するためのdata、split、評価行、後処理条件。 | 条件を具体的に列挙する。 |
| gate | 条件に応じて候補の採用、切替、または重みを決める処理。 | 判定条件と出力を具体的に記録する。 |
| confidence | 予測や候補の不確実性または信頼性を表す特徴や診断値。 | 算出方法を併記する。 |
| port | train側または参照実装の処理をinference codeへ移すこと。 | 移した処理と差分を記録する。 |
| replacement | 既存予測を別候補で置き換えること。 | 対象範囲が全行か一部かを記録する。 |
