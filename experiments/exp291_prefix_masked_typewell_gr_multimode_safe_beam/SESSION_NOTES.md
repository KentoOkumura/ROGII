# exp291_prefix_masked_typewell_gr_multimode_safe_beam セッションノート

## 目的

exp284からsame-well self-GRを完全に外し、visible Type Well GR likelihoodの局所極大を
全件保持するsafe-anchored beam仮説を、実装前に反証可能な固定contractとして確定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU version 1完了・固定guard FAIL・branch closed
- 親: `exp284_prefix_masked_wrong_mode_branch_recovery_backtest`
- CV: known-prefix masked backtest完了（guard FAIL）
- LB: 対象外
- 実装承認: あり（2026-07-19ユーザーメッセージ）
- canonical train notebook採用: あり
- Kaggle push承認: あり（固定CPU run 1回）

## 2026-07-19 実装

### 実装内容

- exp284 compact trainのfold-safe exp226 geometry replay、target-safe loader、SHA utility、
  truth freeze構造を再利用し、wrong-mode injectionとsame-well self-GR proposalを削除した。
- pre-cut 128 rowsで固定13 shift bankをscoreし、`abs(shift) >= 10 ft`かつfull-bank隣接slot以上の
  local maximumを全件保持する。0件ではsafe-onlyとし、強制fallbackを置かない。
- candidate bankはsafe 1本、real local modes全件、eventごとにreal countを保つscore-blind
  matched-count shuffled modesに分けた。real/shuffledの偶然のshift重複は許容する。
- branch pathはexp226 safe geometryの先頭512 rowsに固定shiftを加える定数shift branchとした。
- H128はH128、H256はH128/H256、H512はH128/H256/H512の全checkpointで同一alternativeが
  safeをstrictに上回り、各checkpointでvetoを通った場合だけcommitする。
- policyはsafe-only / visible top1 / all real modes / matched-count shuffleの4件。
- post-cut truthはmask、shift score、candidate、branch、evidence、policyのlogical content SHAを
  固定し、target-free CSVを保存した後にだけ読む。
- pairwise guardはreal alternative対safeのH256 evidence-margin AUCと、H128/H256 persistent
  choiceのbalanced accuracyとして実装した。
- inference aliasは明示的なdisabled reportだけを持ち、prediction/submissionを生成しない。

### notebook構造比較

- 親exp284 compact train: 10章、2427行。
- exp291 compact train: 10章、2365行。
- Imports、runtime/SHA、exp226 geometry、candidate bank、persistent policy、post-freeze readout、
  orchestration、setup、runの各役割を維持し、同一exp helper importだけの薄い構成にはしていない。
- 既存canonical train/inference notebookは、リポジトリ規約に従い採用判断なしに上書きしていない。

## 2026-07-19 Kaggle CPU実行承認

ユーザーの「実行してください」を、別名compact train notebookのcanonical採用と、固定contractの
Kaggle CPU 1回実行の承認として記録する。inference / submissionは承認範囲外で無効のままとする。

- active contract: 1
- fixed policy: 4
- LightGBM / model config: 0
- trained fold: 0
- total booster: 0
- HMM / PF regeneration: 0 / 0
- parent/control再学習: なし
- runtime: Kaggle CPU、GPU/TPU/internet off、single process
- input kernel sources: exp226 train、exp115 hidden-like train
- exp209 emission: notebook内の固定式を使用し、exp209生成物依存なし
- canonical kernel: `kentookumura/exp291-typewell-multimode-safe-beam-backtest-train`
- canonical title: `exp291 typewell multimode safe beam backtest train`
- parameter rescue: 禁止

credential preflightはAPI token未設定、OAuth credentialとlegacy credentialはOK。exp284で長いfull slugが
`SaveKernel 400`になった履歴を踏まえ、意味を保った50文字のshort canonical slugを初回から使用する。

### canonical採用・package監査

compact train notebookをcanonical train notebookへ採用した。両者はbyte-identical、22 cells、code cell
10、cell output 0。専用tests 9/9、Ruff full、strict experiment validationはPASSした。

strict packageはbootstrap 1 + canonical 22 = 23 cells、cell output 0。bootstrap ZIP内configはrepo側と
byte-identicalで、private、CPU、GPU/TPU/internet off、run-on-push true、competition source 1件、
kernel source 2件を確認した。

- executed config SHA: `e6f995a79b8802c0ae49033598d42ca1ce64f0e587b475e24eb1dc66c8bed2ef`
- canonical notebook SHA: `17551b9e688c5ed75144f4b38f8ee7ab223302c2e67feebe85eab9b1373e9014`
- packaged notebook SHA: `8a45bb22597f84d06eec24c56f383abf62818875c858a2fe5018339fa92ee881`
- kernel metadata SHA: `e44abb5bba5032366218b19e7944b2408d4d37a661446b97d027b62e829ef095`
- push前server pull: 403。既存canonical kernelなしと判断した。

### Kaggle CPU version 1 push

2026-07-19 21:07 JSTに初回pushが成功し、version 1を開始した。push直後のserver metadata pullも成功し、
local packageと同じprivate / CPU / GPU・TPU・internet off、competition source 1件、kernel source 2件を
確認した。

- version / id_no: `1 / 127882960`
- URL: `https://www.kaggle.com/code/kentookumura/exp291-typewell-multimode-safe-beam-backtest-train`
- status: `COMPLETE`

### Kaggle CPU version 1 完了

2026-07-19 23:00:58 JSTに固定contractが完了した。runtimeは6,805.497秒（約1時間53分25秒）。
766 eligible / 7 ineligible wells、fold 0〜4を評価し、technical guardは全項目PASSした。
post-cut truth access before freezeは0、self-GR候補は0、candidate/branch/evidence coverageは1.0である。

| H256 policy | RMSE |
| --- | ---: |
| safe-only | 4.827483 |
| top1 Type Well mode | 18.713110 |
| all Type Well modes | 22.199818 |
| matched-count shuffle | 17.360718 |

- all-mode gain vs safe: `-17.372335 ft`、改善fold `0/5`
- all-mode gain vs top1: `-3.486709 ft`、改善fold `1/5`
- H512 gain vs safe: `-11.497241 ft`（H256以上の持続性条件だけPASS）
- safe unique-best false switch: `34.9462%`（上限5%をFAIL）
- pairwise evidence AUC pooled: `0.672737`
- balanced choice accuracy pooled: `0.576907`（下限0.60をFAIL）
- fold AUC: `0.869081 / 0.636179 / 0.563686 / 0.689306 / 0.467049`
- fold balanced accuracy: `0.695543 / 0.527439 / 0.398374 / 0.589234 / 0.501791`
- all-modeはmatched shuffleよりpooledで悪く、非悪化foldは`0/5`

pairwise AUCにはpooled signalが残ったが、alternative better rateは1.2128%にすぎず、固定
persistent commitはsafeを保護できなかった。技術不良ではなくscientific / safety guard FAILと判定し、
contractどおり `close_without_parameter_rescue` とする。decoder、inference、prediction、submissionは
生成しない。

小さいcontract / mask / input / overall / fold / pairwise / summaryだけをKaggle outputから取得し、
summary記載SHAとbyte SHAの一致を確認した。巨大なbranch path archiveは取得していない。

## 2026-07-19 設計判断

### 根拠

- exp280では固定shift bankのType Well GR likelihoodにtop1/top3のshuffled超えがあったが、
  絶対精度は直接shift補正を許す水準ではなかった。
- exp284ではself-GR込みfull policyがself-GRなしpair policyよりH256で2.438300 ft悪化し、
  0/5 folds改善だった。self-GRを候補源から除外する根拠とする。
- exp284のType Well pair evidenceもpooled AUC 0.675153に対しfold 3/4が
  0.509459/0.555936であり、early top1 commitは安全でない。
- したがって「all modesを残す」だけでなく、safe絶対保持と複数checkpointの継続優位を
  commit条件に含める。

### 固定contract

- mask 640 rows、visible 512 rows以上、pre-cut score 128 rows、H128/H256/H512、primary H256。
- shift bankは `[-80, -40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40, 80] ft`。
- alternativesは `abs(shift) >= 10 ft` のType Well GR local maxima全件。
- local maximumはfull bankの隣接score以上、端点は片側比較。該当なしならsafe-only。
- safe baseは常に候補に残し、prune、blend、平均化しない。
- self-GR、NCC、donor window/orientation、self-GR shuffleは使用しない。
- H256 commitは同一modeがH128/H256の両方でsafeをstrictに上回り、veto通過した場合だけ。
- 複数modeがcommit条件を満たす場合はH256 evidence最大、同点はshift bank順で選ぶ。
- tie、nonfinite、veto失敗、候補なしはsafe。
- policyはsafe-only、top1、all modes、matched-count shuffleの4件。shuffleはreal候補数を保ち、
  eligible nonzero bankからscore-blind・重複なしでstable local RNG抽出する。
- truthはtarget-free artifactsのcontent SHA freeze後だけjoinする。

### Kaggle train push前の計数（設計値）

- active contract: 1
- fixed policy: 4
- LightGBM config: 0
- trained folds: 0
- total boosters: 0
- HMM/PF regenerations: 0
- parent/control retraining: なし
- GPU: なし、CPU single process予定

親/control再学習を含まない。これは設計時点の計数であり、後続のユーザー承認後に同じ固定計数で
Kaggle CPU version 1を実行した。

## guard

- technical: eligible 750 wells以上、5 folds、mask/safe/all-mode/finite coverage 1.0、
  pre-freeze truth access 0、self-GR candidate 0。
- evidence: safe-vs-alternative AUC各fold 0.60以上、balanced choice accuracy pooled 0.60以上、
  各fold 0.50超。
- primary: all-mode H256がsafe比0.10 ft以上、4/5 folds以上改善。
- multi-mode value: all-modeがtop1比0.05 ft以上、3/5 folds以上改善。
- safety: safe unique-best false switch 5%以下、H512 gainがH256を下回らない。
- negative control: matched-count shuffleよりpooled改善、5/5 folds非悪化。
- failure: 1つでも失敗すればparameter rescueなしでclose。

## コマンドログ

2026-07-19に設計scaffoldだけを作成した。

```bash
make new-steering EXP=exp291_prefix_masked_typewell_gr_multimode_safe_beam
make new-exp EXP=exp291_prefix_masked_typewell_gr_multimode_safe_beam
```

実装・静的検証で実行した主なコマンド:

```bash
.venv/bin/pytest -q tests/test_exp291_prefix_masked_typewell_gr_multimode_safe_beam.py
.venv/bin/ruff check <exp291 train/inference/test>
.venv/bin/python -m py_compile <exp291 train/inference>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <exp291 compact source>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <exp291 compact source>
make validate-template
make validate-exp EXP=exp291_prefix_masked_typewell_gr_multimode_safe_beam
make test
```

- exp291 dedicated tests: 9 passed。
- repository tests: 260 passed（最終再実行で確認）。
- Ruff / py_compile / Jupytext round-trip / strict experiment validation: passed。
- `task validate-exp` は環境に`task`がないためexit 127。同等の`make validate-exp`はpassed。
- Kaggle CPU train version 1を実行した。LightGBM学習、推論、提出は未実行。

完了確認と限定output取得:

```bash
kaggle kernels status kentookumura/exp291-typewell-multimode-safe-beam-backtest-train
kaggle kernels logs kentookumura/exp291-typewell-multimode-safe-beam-backtest-train
kaggle kernels output kentookumura/exp291-typewell-multimode-safe-beam-backtest-train \
  -p /tmp/exp291-v1-metrics \
  --file-pattern '.*(summary\.json|overall_metrics\.csv|fold_metrics\.csv|pairwise_metrics\.csv|input_manifest\.csv|mask_manifest\.csv|contract\.json)$'
```

## 再現性メモ

- seed policy: 実modeはdeterministic、shuffle controlのみstable SHA256 per well/cut/source local RNG
- stochastic components: matched-count shuffled modes negative controlのみ
- CPU/GPU runtime: CPU worker 1予定、GPU/AMP off
- Kaggle kernel id / version: `kentookumura/exp291-typewell-multimode-safe-beam-backtest-train` / `1`
- executed config SHA: `e6f995a79b8802c0ae49033598d42ca1ce64f0e587b475e24eb1dc66c8bed2ef`
- target-free content SHA:
  - shift score: `5e2b6cf1a34b95495b2f6e2a6c854fecbe9de4a8682d43dff503988ee774c849`
  - candidate: `07bea4d3dceb6fa4404c4ca79df68c6ed8cdee46f275e74650ff8471a842cf3a`
  - branch: `2fd84727ed795628c942703aa6ab897f13565855005e753d19f36932f62a8b4c`
  - evidence: `84d4d5e40e1c0a57dc02357b5d46f2c6cfb476571613980982e189d0833ba0e5`
  - policy: `d914824e96159eeeab11028b54392507d2c4dbbca8b17656a2986d04b934bb17`
- model manifest / model SHA: model非生成
- prediction SHA: prediction非生成
- submission SHA: submission非生成
- rerun check: 未実行（固定contract version 1を正とする）

## 次のアクション

1. exp291 branchをclosedのまま維持する。
2. 同じbacktest truthでK、window、shift、horizon、margin、likelihood、vetoを救済調整しない。
3. exp291からdecoder、inference、prediction、submissionへ進めない。
