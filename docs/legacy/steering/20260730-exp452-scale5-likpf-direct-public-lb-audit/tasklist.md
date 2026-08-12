# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- Codexによる再提出、追加run、LB適応は行わない。

## 完了

- exp452を採番し、標準experiment scaffoldを作成した。
- 候補を`likpf_scale_5_x1p0`の1本に固定した。
- evaluation/inference Notebookを1本とする契約を固定した。
- PF、aggregation、seed、parity、hidden cardinality、LB解釈、禁止事項を固定した。
- 再現性設計を`design.md`と`config.yaml`へ記録した。
- 2026-07-31のユーザー承認に基づき、Jupytext percent形式のcompact
  self-contained inference sourceを実装し、正規inference Notebookへ採用した。
- exp413 v4が使ったexp073 PF source SHA `4af212...`から`stable_seed`、
  `_interp1`、`_grid`、`_pf_lik_allseeds`を抽出し、AST parity testを追加した。
- dynamic sample/ID/nonempty-well、finite 100%、fallback 0、1候補限定、
  実行量、input/schema/content/prediction/submission SHA guardを実装した。
- 公開3 wells・14,151 rowsのopt-in function testで500 particles × 128 seedsを
  再生成し、exp413 v4 `likpf_scale_5`とfloat32最大差`0.0 ft`、logical content
  SHA `b713ade7...`を確認した。Notebook自体のローカル実行は行っていない。
- Jupytext test、py_compile、Ruff F821、専用test、`make validate-exp`を通した。
- package metadataとbootstrap内config/source SHAを監査し、private CPU、internet off、
  `competition_submission_approved: false`を確認した。
- 52文字slugの初回pushがKaggle `SaveKernel 400`で実行前に拒否されたため、
  科学条件を変えず42文字のcanonical slugへ短縮した。
- private Kaggle CPU inference version 1（id_no `129271895`）を完走した。
- 14,151 rows / 3 wells、500 particles × 128 seeds、fallback 0、公開参照との
  float32最大差`0.0 ft`、prediction content SHA `b713ade7...`を確認した。
- outputを`/tmp`へ取得し、sampleとのheader / 行数 / ID順序、finite、duplicate、
  prediction/submission差、全manifest/file SHAを確認した。submit-checkは
  FAIL 0 / WARN 0でPASSした。
- Codexはcompetition submissionを開始していない。
- ユーザー外部提出ref `55149125`がexp452であることを確認し、Kaggle
  `COMPLETE` / Public LB `8.797`を記録した。
- exp434 v10 SHA256 arithmetic LikPF `9.807`より`1.010`改善し、OOFと方向一致した。
- exp417のby-well tail FAILを維持し、自動昇格、再提出、救済変更を禁止した。

## 次のアクション

- なし。Public LB `8.797`を記述censusとして閉じ、追加runや再提出を行わない。
