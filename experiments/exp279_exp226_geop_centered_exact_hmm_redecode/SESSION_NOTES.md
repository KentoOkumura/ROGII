# exp279_exp226_geop_centered_exact_hmm_redecode セッションノート

## 目的

group-safe exp226 `tvt_geop`を毎行の絶対基準としてexp209 exact HMMをjoint re-decodeし、
GR mode slip後の持続offsetに弱い復元力が効くかを1 fixed variantで反証可能に検証する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU train version 1完了、promotion guard FAIL、branch closed
- CV / LB: 10.035987 / 未実行
- inference / submission: disabled

## 実行コスト契約

- active HMM variant / well-runs: `1 / 773`
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- parent/control再学習・再生成: 0
- outer workers / Numba threads: `1 / 4`
- runtime: Kaggle CPU、GPU off、internet off
- exp221実測を基準に約5時間を想定

## 変更点

- `docs/legacy/steering/20260718-exp279-exp226-geop-centered-exact-hmm-redecode/`を作成。
- `backlog/KAGGLE_DIRECTION.md`の「実装済み・Kaggle train待ち」にbacklogを追加。
- exp270 exact `_hmm2_fb`とAST一致するkernelをself-contained train sourceへ展開。
- exp209固定grammarへexp226 `tvt_geop` Gaussian unary `sigma=20 / lambda=0.50 / clip=600`を追加。
- exp226 `tvt_pred` / `gr_delta` / truth/error列をdecoder入力から除外。
- exp226 / exp209 / exp072 gzipのdecompressed SHAをhard guardし、exp263 fixed formulaを保存予測だけで再構成。
- 再構成exp263 fixedのOOF RMSEを保存基準8.238331へ`1e-5 ft`でparityする。
- overall / fold / distance / hidden-like / by-well / persistent-offset recovery / promotion guard / SHA保存を実装。
- inference notebookはtrain guard通過とユーザー承認までfail-closedのdisabled contractとした。

## 事前read-only監査

- exp226 OOF: 3,783,989 rows / 773 wells。
- exp209固定grid内の`tvt_geop`: 3,783,989 / 3,783,989 rows（100%）。
- 全評価行がgrid内なので、grid拡張・residual-grid化は行わず単純unary設計を維持した。

## Notebook構造比較

- 実装参照exp270 train source: 2,309行 / Contents + 10 numbered sections。
- exp279 train source: 1,401行 / Contents + 9 numbered sections。
- exp279はtop-K decoder、2-shard aggregate、oracle bankを除き、代わりにexp226 input契約、
  Gaussian unary、exp263 control再構成、復帰診断、promotion guardをセル上へ展開した。
- 同一exp helper importなし。train/inference sourceの`__file__`依存なし。

## コマンドログ

```bash
make new-steering EXP=exp279_exp226_geop_centered_exact_hmm_redecode
make new-exp EXP=exp279_exp226_geop_centered_exact_hmm_redecode
.venv/bin/python -m py_compile <exp279 train.py> <exp279 inference.py> <exp279 test.py>
.venv/bin/ruff check <exp279 sources and test> --select F821,F401,F841,E722
.venv/bin/pytest -q experiments/exp279_exp226_geop_centered_exact_hmm_redecode/tests/test_exp279_exp226_geop_centered_exact_hmm_redecode.py
# 4 passed
make validate-exp EXP=exp279_exp226_geop_centered_exact_hmm_redecode
# strict validation passed
make validate-template
make test
# repository 161 tests passed
```

この事前実装・静的検証の時点では、ローカルfull notebook、親HMM再生成、Kaggle prepare/pushは
実行していない。後続の承認・push・結果は以下の時系列記録を正とする。

## 再現性メモ

- RNG: なし。well文字列昇順、exp226保存fold。
- input primary SHA: exp226 `709eb726...e4c609`、exp209 `ee3b548b...2ee3f4`、
  exp072 `99a3c70a...0e1350`（いずれもdecompressed content）。
- decoder manifest: HMM、unary、fixed formula、truth attachment contractをSHA化する。
- output: raw/decompressed prediction gzip SHA、logical prediction content SHA、全CSV/JSON SHAを保存する。
- model SHA: fitted modelなしのためdecoder manifest SHAで代替。
- submission SHA: inference/submission無効のため対象外。
- deterministic anchor: Kaggle実行・rerun前はfalse。

## 最終判断

1. Kaggle CPU version 1の結果とSHA監査を実験記録へ反映済み。
2. guard FAILのためsigma/lambda救済やinferenceへ進まずbranchを閉じた。
3. 完了済みexp279をbacklogから削除し、同一unaryの直接救済backlogは追加しない。
4. 後続のユーザー指示により、修正版exp264の既存12候補を維持したまま`geop_hmm`を疎な
   13番目候補として評価する別仮説を`backlog/KAGGLE_DIRECTION.md`へ追加した。これはexp279の再開ではない。

## 2026-07-18 Kaggle CPU実行承認

- 承認時刻: `2026-07-18 19:25:15 JST`
- 承認scope: canonical private train kernelを1回pushし、完了まで監視する。
- 実行量: 1 active HMM variant / 773 well-runs / 0 LightGBM config / 0 trained fold / 0 booster。
- parent/control再学習・再生成: 0。exp226 / exp209 / exp072の保存済み予測だけをcontrolに使う。
- runtime: Kaggle CPU、outer workers 1、Numba threads 4、GPU/TPU/internet off。
- inference / submission: disabled。本承認にはraw-test portと提出を含めない。
- credential: OAuth CLIとlegacy CLI credentialを確認。API tokenは未設定だがKaggle CLI操作には影響しない。
- canonical kernel: `kentookumura/exp279-exp226-geop-exact-hmm-redecode-train`。
- remote preflight: canonical pullは403、`kernels list --search exp279 --mine`は`Not found`。
  既存kernelは確認されず、別slugを増やさず初回canonical pushへ進む。

## 2026-07-18 Kaggle CPU version 1 package

- config source/package/bootstrap SHA:
  `60222beae69940e04ecca50984a6b109bef36b452100fd9135d7d2397024241b`。
- train source/package/bootstrap SHA:
  `2acdcc57066cae04225aa6e42bb10faae26cc1cb8d92473f3256448d9454df85`。
- prepared notebook SHA:
  `de03b6f0fcf8ce86b1081b673ac5715494bae86041770b9ddacbe3c96631602d`。
- bootstrap files: 15。config / train sourceはsourceとbyte一致。
- metadata: canonical id/title slug一致、private、CPU、GPU/TPU/internet off、run-on-push true、
  competition source 1、kernel sources 4。
- package configは`kaggle_push_approved=true`、1 variant / 773 well-runs / 0 config / 0 fold /
  0 booster / parent再生成0 / inference off / submission off。

## 2026-07-18 Kaggle CPU version 1 push

- kernel: `kentookumura/exp279-exp226-geop-exact-hmm-redecode-train`
- version: 1
- push成功後、local `execution.kaggle_push_approved=false`へ戻しversion 2の誤pushを防止した。
- URL: `https://www.kaggle.com/code/kentookumura/exp279-exp226-geop-exact-hmm-redecode-train`
- Kaggle `id_no`: `127766774`。
- push後pullでcanonical id/title、private、CPU (`machine_shape=None`)、GPU/TPU/internet off、
  competition source 1、kernel sources 4を確認した。
- Kaggle正規化後notebook SHA:
  `8b8919ff5f8829f943add8313dfb22d06e807a72cd1070d7e6ded970e34fcf22`。
- 60秒間隔のmonitorを開始し、完了時のoutput取得先を
  `/tmp/kaggle-output/exp279_exp226_geop_centered_exact_hmm_redecode/train_v1`に固定した。
- 初期statusは`KernelWorkerStatus.RUNNING`。通常logsは空だが、platform既知挙動のため失敗判定しない。

## 2026-07-19 Kaggle CPU version 1 result

- status: `KernelWorkerStatus.COMPLETE`。
- runtime: `18,663.388503`秒（約5時間11分）。
- contract: 1 HMM variant / 773 well-runs / 0 LightGBM config / 0 trained fold / 0 booster /
  parent-control再学習0 / GPU off / internet off / inference off / submission off。
- coverage: 3,783,989 / 3,783,989 rows、773 / 773 wells、全well status `ok`、
  geometry grid / finite coverage 100%。
- package config / train source SHAはpush前記録
  `60222be...4241b` / `2acdcc...4df85`と取得outputがbyte一致した。
- input decompressed SHAはexp226 `709eb726...e4c609`、exp209 `ee3b548b...2ee3f4`、
  exp072 `99a3c70a...0e1350`と一致した。

### Performance

- exp226 prediction RMSE: `9.4271095966`。
- exp209 exact HMM RMSE: `11.9382872349`。
- exp263 fixed baseline RMSE: `8.2383317455`。保存基準との差`7.45e-7 ft`でparity PASS。
- `geop_hmm` RMSE: `10.0359869413`。
- exact HMM比では`-1.9023002936 ft`改善したが、promotion baseline exp263比は
  `+1.7976551959 ft`悪化し、overall gain guard FAIL。
- fold delta vs exp263: `+2.848074 / +1.593594 / +2.103474 / +1.272560 / +1.248048 ft`。
  改善0 / 5 foldsでfold guard FAIL。
- near / 1000+ delta: `+0.198846 / +1.983953 ft`。
- hidden-like spatial / typewell-purged delta: `+1.381884 / +1.451614 ft`。
- worst-well delta: well `389ae58f`で`+27.158481 ft`。
- well単位: 267改善 / 506悪化、delta中央値`+0.971711 ft`。
- technical guard 3件はPASS、performance guard 4件は全FAIL、総合promotion guard FAIL。

### Persistent-offset recovery

- exp263 fixed: 551 episodes、256 / 512行以内復帰`2.1779% / 9.0744%`。
- `geop_hmm`: 802 episodes、256 / 512行以内復帰`2.6185% / 11.8454%`。
- 復帰率は小幅上昇したがepisode自体が251件増えたため、持続offsetを安全に抑える仮説は
  overall結果として支持されない。

### SHA audit

- summary記載のby-well / candidate / decoder / distance / episodes / fold / hidden / input /
  OOF raw gzip / recovery / well manifestの11 SHAを取得fileと全件照合しPASS。
- OOF raw gzip SHA:
  `2f94f4977004d35c3ce443dffe241d3500a250042ece120b05114dc795732d6b`。
- OOF decompressed SHA:
  `9e29e70783eb13abff5ccbee23acbb274589457f233fdb266094a4db0a24d7c0`。
- logical prediction SHA:
  `335ba03183d5c18140d237a2d9adf7a39bb5fdb5dd00c56c507dcd59bc9e613c`。
- decoder scientific manifest SHA:
  `0a3af93a14376925e08340265fb42a6f694203ac5fc131087bb4675f22ecc021`。
- decoder manifest file SHA:
  `b02eeee8466e4af567936da89fc63d6f467bd73ca78af65f5c383710910e63fc`。
- summary file SHA:
  `b4332b3c72da37447bfc45cf4b974a45b4571682a46346da671374fed6e9ae87`。
- 成功runはversion 1の1回だけなのでdeterministic anchorとは呼ばない。

### Decision

`completed_train_side_guard_failed_branch_closed`。geometry unaryはexact HMM単体を改善したが、
exp263より全foldと全保護scopeで回帰し、persistent-offset episodeも増えた。事前契約どおり
sigma / lambda / grid / process-noise救済、PF併用、blend / selector、raw-test inference、
submissionを行わない。完了済みbacklogを削除し、同一unaryの直接救済backlogは追加しない。
後続のユーザー指示によるexp264 add-only selector候補化は、別仮説・別backlogとして管理する。
