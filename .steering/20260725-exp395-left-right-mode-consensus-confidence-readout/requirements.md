# exp395 要件

## 依頼

物理モデルが区間ごとに異なる地層対応 mode へ入っている可能性を、予測の補正や
mode 切替を行わずに診断する。

同一の固定 mode identity に対して heel 側と toe 側の重ならない GR evidence を
別々に評価し、両方向が同じ mode を支持するかを confidence として保存する。

このターンでは次だけを行う。

- `KAGGLE_DIRECTION.md` の backlog へ追加する。
- steering を作成して設計契約を固定する。
- design-only の実験 scaffold を作成する。

実装、Notebook の作り替え、Kaggle package、push、run、inference、submission は
行わない。

## 制約

- Route: `pf_beam`
- 親実験: `exp391_prefix_anchored_mode_persistence_hmm_readout`
- decoder 基準: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- exp391 Stage A1 が全 gate を PASS するまで実装へ進まない。
- 2026-07-25: exp391 Stage A1がFAILしたため、本要件に従い未実装で閉鎖した。
- exp386 の空 scenario bank と、閉鎖済み exp387 は再利用・再開しない。
- mode identity は exp391 の transition-overlap lineage を使い、
  top1 / top2 の mass 順位を identity として扱わない。
- 左右 evidence は同じ GR row を共有しない。既知 prefix から得る affine calibration、
  transition grammar、absolute-TVT grid は共通条件として固定する。
- confidence は suffix truth、error、hidden-like role を読む前に freeze する。
- primary confidence は左右 evidence だけから作る。exp226 geometry や
  LikPF / HMM / exp226 / exp263 のモデル間一致は二次 readout とし、
  primary score の重みには混ぜない。
- TVT candidate の生成、予測値の置換、hard mode switch、blend、selector、
  fallback tuning を禁止する。
- 同一 OOF で window、gap、mode threshold、confidence formula、gate を調整しない。
- `docs/06_reproducibility.md` に従い、入力・mode ledger・confidence table・
  scientific contract の logical/content SHA を記録する。

## 受け入れ基準

- primary scope、左右 window、gap、mode matching、confidence formula、
  negative control、late truth join、scientific gate が事前固定されている。
- Stage 0 と full OOF の実行量、依存関係、別承認境界が明記されている。
- confidence-only 診断であり、RMSE 改善や Public LB 6.5を結果として
  先取りしていない。
- 実装・Kaggle実行・推論・提出のフラグがすべて無効である。
- deterministic anchor として扱うのは、同一契約の再実行で logical SHA が
  一致した後だけである。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく
  decompressed content SHA を主証拠にする。
