# 要件

## 依頼

exp295のstructured trainingを坑井全体ではなく固定長windowへ限定する別案を、local CE案とは別expとして設計する。初回依頼ではbacklog、steering、実験scaffoldと設計確定までとし、実装、Notebook実行、Kaggle push、推論、提出は行わない。

2026-07-22の追加依頼「exp332を実装してください」により、exp331のStage A科学gate FAIL・branch closeを先行条件成立として確認し、compact self-contained train/inference候補、専用test、設定・記録の実装までを承認範囲へ追加する。Kaggle package/push、Stage 0 T4実行、Stage A学習、推論、提出は引き続き別承認とする。

2026-07-22の追加依頼「実行してください」により、固定16-window T4 Stage 0のcanonical train採用、package、Kaggle push、完了確認、gate判定、report SHA確認までを承認範囲へ追加する。Stage A fold 0、Stage B/C、推論、提出は承認範囲外のままとする。

## 仮説

exp295のGaussian soft-label structured objective自体を維持しつつ、1 epochのstructured DPを`8,571,405` suffix rowsから約`427,008` window rowsへ固定削減すれば、window内のtransition-awareなunary学習を残しながらKaggle時間内で精度仮説を評価できる。

## 制約

- Route: `ensemble`。
- 親実験: `exp295_prefix_anchored_wholewell_gr_alignment_ssm`。
- window長は256 row、各outer-train well・各epochで3 scheduled slots（最大3 active windows）に固定する。non-overlap windowを確保できないslotはinactive zero-lossとし、重複windowを作らない。window length/count/gridは探索しない。
- objectiveはexp295 version 3と同じGaussian soft-label structured NLL`1.0`（sigma`0.35 ft`）+ local CE`0.25`。
- interior windowはencoderへofficial prefix以外のtruthを入力せず、loss初期化専用のteacher boundaryだけを使う。valid/testはofficial prefixからfull-well decodeする。
- window manifestはtruth/errorを使わず、suffix row範囲とstable SHA256で学習前にfreezeする。
- exp295のinput、fold、architecture、preprocessing、exp209 decoder、controls、promotion gateを固定する。
- 固定16-window T4 microbenchmarkで8.5時間/14 GB gateを通る場合だけfull Stage A候補とする。
- exp331と同時実装・同時GPU比較はしない。原則exp331を先行し、exp331をcloseまたは明示skipした後に別承認で再開する。
- window境界、window数、loss weight/sigma、view/epoch、architecture/band/temperature grid、parent/control再学習は禁止する。

## 受け入れ基準

- `docs/legacy/steering/20260721-exp332-prefix-gr-unary-fixed-window-structured-ssm/`にwindow manifest、boundary supervision、数理objective、計算量、full-well評価、Stage A/B/Cが固定されている。
- `experiments/exp332_prefix_gr_unary_fixed_window_structured_ssm/`にcompact self-contained実装が存在し、未実行・Stage 0別承認待ちが明記されている。
- 1 epochのactive window数は最大`556 wells × 3 = 1,668`、score positionsは最大`427,008`と固定される。
- Stage Aは1 architecture / fold 0 / seed 42 / neural model 1、他model/booster/PF/Beam/control再学習0。
- `make validate-exp EXP=exp332_prefix_gr_unary_fixed_window_structured_ssm`がstrictで通る。
- compact self-contained train候補がwindow schedule、teacher boundary、window内4-sweep structured objective、固定16-window Stage 0、full-well Stage A評価を実装し、`implementation_only`ではGPU処理を開始しない。
- compact self-contained inference候補がStage B promotionと別承認までfail-closedである。
- window選択、truth非参照、boundary非入力、structured gradient、full-well評価契約を専用testで検証できる。
