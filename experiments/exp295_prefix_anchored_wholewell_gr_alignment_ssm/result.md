# exp295_prefix_anchored_wholewell_gr_alignment_ssm 結果

## 状態

Stage A compact self-contained trainをcanonical Notebookへ採用した。Kaggle version 2はhard truth path infeasibleでepoch完了前にruntime failure。fixed decoderを維持したGaussian soft-label structured likelihoodへ修復したversion 3も、epoch 1 summary前にKaggle runtime timeoutとなった。学習完了model、Stage B/C、推論、提出は0。

## 仮説

対象井自身のcomplete-well GRと対応Type Well GRからprefix-conditioned continuous TVT unaryを学び、固定したexact state-space grammarでsuffix全体をposterior推論すれば、neighbor well dataとcandidate selectorなしでpooled OOF 6.0 ft以下、stretch 5.0 ft以下へ到達できる。

## 親実験との差分

exp202のlocal window heatmap/top-K path出力を直接再利用せず、complete-well neural unaryへ置き換える。exp209のstate grid/transitionは固定し、exp221のようなLGB point emissionも加えない。出力はcandidate選択やMTP pathではなく、learned GR unaryだけを観測とするcontinuous TVT posterior meanである。

## 変更点

- local-window/path-headからcomplete-well continuous unaryへ変更する。
- hand-crafted GR scoreからouter-fold neural structured emissionへ変更する。
- candidate selectionではなくfixed state-space posterior meanを直接評価する。
- hard quantized truth path NLLを、`sigma=0.35 ft`のGaussian label observationで条件付けたsoft structured NLLへ修復する。decoder/state grid/transitionは変更しない。

## 設定

- primary parent: `exp202_heatmap_mdn_candidate_generator_probe`
- transition parent: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- decoder reference: `exp221_lgb_oof_gaussian_emission_hmm_on_exp148`
- Route: `ensemble`
- validation: complete-well 5-fold GroupKFold、official hidden suffix全row
- output: exact forward-backward TVT posterior mean
- neighbor well data / candidate bank / hard top1 / existing model blend: すべてなし
- seed: 42 + stable SHA256 per well/fold/pseudo-cut/control

## 結果

| メトリック | 値 |
| --- | --- |
| Stage A fold 0 | version 2 objective failure、version 3 runtime timeout（model未学習） |
| Stage B pooled OOF | 未実行 |
| GR attribution | 未実行 |
| Public / Private LB | 未提出 |

## 固定判定

- Stage Aは1 architecture / 1 fold / 1 seed / 1 neural modelだけを許可する。
- Stage BはStage A全PASS後、fold 0を再学習せずfold 1-4の4 modelsだけを追加する。
- pooled OOF `<=6.0 ft`、5/5 folds、real-vs-controls、1000+、hidden-like、well p95/worstを全PASSした場合だけStage C候補とする。
- `6.0 < OOF <=6.75 ft`かつGR attribution PASSは別expのarchitecture iteration根拠に限定する。
- `OOF >6.75 ft`、GR attribution FAIL、またはStage A FAILではbranchを閉じる。

## 再現性

- deterministic anchor: false
- kernel: `kentookumura/exp295-prefix-gr-ssm-fold0-train` version 3 `CANCEL_ACKNOWLEDGED`（version 2 `ERROR`）
- fold/pseudo-cut/input/feature/model/emission/posterior/prediction SHA: 未生成
- submission SHA: 対象外
- rerun result: epoch/model/生成物0でruntime timeout

## 解釈

数値結果はない。neighbor-free boundary、complete-well learned emission、fixed SSM、posterior mean、truth-late readoutを実装したが、1,668 fit views・8,571,405 suffix rowsに対する4-sweep exact DPは1 epochをKaggle時間内に完了できなかった。仮説の精度は未評価で、実装可能性のruntime guardだけがFAILした。

## 次

固定Stage A runtime gate FAILとしてexp295をbranch closeする。再訪はlocal CE trainingまたはfixed-window structured trainingを別expとして事前設計する場合だけ検討する。
