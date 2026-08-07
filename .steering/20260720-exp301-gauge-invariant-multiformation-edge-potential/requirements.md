# 要件

## 依頼（2026-07-20 設計時）

- 案1 `gauge_invariant_multiformation_edge_potential` を `exp301` としてバックログ、steering、実験ディレクトリへ登録し、実装前の設計を確定する。
- 今回は solver、特徴量生成、学習、推論、Kaggle push、submission を実装・実行しない。
- 案2と案3は、別セッションでも内容や開始条件が変わらないよう、案1配下の固定契約として保存する。

## 現在の依頼（2026-07-20 実装）

- ユーザーの「exp301を実装してください」を実装承認として、Stage 0 / 条件付きStage 1のコードとtestsを作成する。
- Stage 0 FAIL時は同じrunでStage 1を実行せず、branchをfail-closedで閉じる。
- Kaggle prepare/push、ローカルnotebook実行、inference、submission、案2/案3は今回の依頼に含めない。
- 既存の正規notebook placeholderは上書きせず、compact self-contained候補を別名で生成する。

## 解く問題

train horizontal の6 formation列は絶対値にwell固有datumを含む一方、同一well内の差分ではdatumが消える。
そこで絶対formation値や既存predictionを回帰するのではなく、6 formationの井戸内edge差分だけを観測として、
2次元の共通スカラーポテンシャルを復元する。known prefix末尾をanchorにして、未知suffixのTVTを単一の
direct physical candidateとして生成できるかを検証する。

## 制約

- Route: `pf_beam`。
- 親実験なしの独立physical familyとし、exp289のfault-risk gateやANCC-only solverの救済実験にしない。
- outer-valid/testのformation 6列とtrue TVTはsolver入力、scale推定、inner選択、grid構築へ一切入れない。
- outer-valid/testから使用できるのは`well_id`, row identity, `MD`, `X`, `Y`, `Z`, `TVT_input`だけ。GRも使わない。
- outer-valid foldはtarget wellだけでなくfold全体をdonor constraint、inner選択、robust scale推定から除外する。
- current testと同名のtrain wellはinference donorから除外する。
- formation絶対値、typewell top、well datum、既存OOF prediction、candidate bank、ML、GR、fault cutをprimary solverへ入れない。
- 予測は1本のdirect candidateだけ。blend、selector、oracle predictionの保存、posthoc補正は禁止する。
- fixed numeric contractは`design.md`とexperiment `config.yaml`を正とし、outer-valid truthに合わせたgrid/stride/lambda/threshold救済を行わない。
- 再現性は`docs/06_reproducibility.md`に従う。gzip artifactはraw SHAに加えてdecompressed content SHAを主証拠にする。

## 受け入れ基準（実装）

- steering 3文書、標準experiment scaffold、`config.yaml`、設計専用fail-closed notebook、案2/案3の固定契約が存在する。
- `KAGGLE_DIRECTION.md`の未着手バックログと`experiment_summary.md`にexp301が登録される。
- Stage 0 / Stage 1、leakage境界、数値契約、technical/scientific gate、failure policyが実装前に固定される。
- 実装承認とKaggle実行承認がともにfalseで、notebookを誤実行してもsolverやpredictionを生成しない。
- 案2/案3の正は`experiments/exp301_gauge_invariant_multiformation_edge_potential/reserved_followup_contract.md`であると、steering、README、configの全てから参照できる。
- Jupytext percent形式のcompact self-contained train sourceが、safe loader、Stage 0、固定solver、prediction freeze、late truth join、exp226比較、exp293 add-one H512 novelty、SHA保存を実装する。
- gauge shift、affine recovery、formation permutation、valid poison、fold exclusion、same-name exclusion、no-donor component、stable SHAのunit testsが存在しPASSする。
- `config.yaml`は実装承認true、Kaggle実行/inference/submission falseを維持する。
