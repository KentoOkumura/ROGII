# 設計

## アプローチ

exp293/exp302 と同じ固定12候補バンクを exp263 の保存済み candidate partition
から再構成し、content SHA を照合する。exp333 Stage 1 OOF は
`well_id,row_idx,outer_fold,tvt_pred_stage1` だけを先に読み、
固定バンクの行へ厳密に整列する。H128/H256/H512/whole-well の block
assignment と合わせて SHA freeze した後にだけ evaluation truth を読み、
固定12候補 oracle と exp333 add-one oracle の RMSE 差、strict unique-best
fraction、fold 安定性を算出する。

## 実験範囲

- 対象実験: `exp361_exp333_candidate_path_addone_novelty_audit`
- Route: `ensemble`
- 親実験: `exp333_exp226_k16_segment_residual_offset_target`
- 変更する変数: 固定12候補バンクに exp333 Stage 1 OOF を1本だけ add-one する。
- 固定する変数: exp293 candidate order/SHA、exp302 block contract と閾値、
  exp226 evaluation folds、float32 cast、tie policy、truth join 順序。
- 参考比較: exp226 direct control、exp228 row-wise residual ablation、
  exp263 fixed blend。いずれも novelty hard gate にはしない。
- 実行量: 1 candidate × 5 reporting folds、model config 0、trained fold 0、
  booster 0、parent/control regeneration 0、GPU 0。

## 再現性設計

- seed policy: 新しい乱数を使わず、保存済み exp226/exp333 outer-fold identity を使う。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。保存済み固定候補値だけを読む。
- 並列処理と乱数の関係: 単一 Python process、乱数なし。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU false、internet false。
- train cache / test feature regeneration の SHA 記録方針: exp263 manifest、
  candidate bank content、block decompressed content、exp333 OOF decompressed
  content と logical prediction SHA を記録する。test feature は生成しない。
- model manifest / prediction / submission SHA 記録方針: 新規 model/submission
  はない。freeze 後の exp333 aligned prediction と全出力の SHA manifest を保存する。
- Kaggle package bootstrap 確認方針: numpy/pandas/PyYAML の既存 runtime
  だけを使い、internet bootstrap を行わない。kernel metadata と config を strict 検証する。

## リスク

- リークリスク: exp333 OOF に `tvt_true` が同居するため、pre-freeze は
  `usecols` allowlist で真値を開かず、truth access count 0 を契約化する。
- CV/LB 不一致リスク: oracle headroom は selector が実現できる性能ではない。
  PASS は推論候補生成の価値だけを示し、LB 改善や採用を保証しない。
- ランタイム/メモリリスク: 約378万行×12候補を float32 memmap で保持する。
  exp302 と同じ chunk/SHA 実装を再利用し、CPU 1 process とする。
- 再現性リスク: Kaggle notebook input の slug/path が変わり得るため複数 pattern
  を許す一方、file/decompressed/post-read content SHA と canonical ID SHA を hard check する。
  ただしexp333学習時のin-memory pandas row-hashはCSV round-trip後のhard checkにせず
  upstream evidenceとして記録し、post-read predictionにはexp361側のcontent SHAを付ける。
