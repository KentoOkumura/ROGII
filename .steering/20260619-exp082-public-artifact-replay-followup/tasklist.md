# タスクリスト

## TODO

- `jaemin_sp45_fleongg_final` は fle3n final と近いため代替候補として保持する。
- `rauffauzanrambe` は direct-output reference として保持し、code-submit 再現候補にはしない。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- `exp079_public_artifact_replay_integrity_audit` から `exp082_public_artifact_replay_followup` を作成した。
- SP45 / fle3n / Koolbox / SP45-Fleongg blend の exact source slug を `config.yaml` に固定した。
- `.ipynb` と `.py` source の risk inspection を実装した。
- Kaggle train notebook を `kentookumura/exp082-artifact-followup-train` v1 として実行し、audit output を取得した。
- `SESSION_NOTES.md`、`result.md`、`metrics.json`、`KAGGLE_DIRECTION.md` を実測値で更新した。
- `rauffauzanrambe/rogii-sp45-wellbore-for-blend-prediction` の existence、direct output availability、source addability probe を確認した。
- `rauffauzanrambe` を required source から外し、mountable source だけで `kentookumura/exp082-artifact-followup-train` v2 を再実行した。
- v2 で `audit_completed`、missing required sources 0 を確認した。
- fle3n / jaemin / rauff direct-output の SP45 projection 3 件を submit-check し、すべて FAIL/WARN なしで sample 互換を確認した。
- `sp45_projection_candidate_guard.py` で row-level guard を実行し、候補 SHA、anchor distance、候補間 distance を `artifacts/` に保存した。
- `sp45_fleongg_source_port_next_candidates.py` を追加し、fle3n final / jaemin final / Pilkwang branch shortlist の source-port next-candidate guard を実行した。
- `fle3n_final_blend` と `jaemin_sp45_fleongg_final` は archived source と source risk 上は hidden-compatible source-port run 候補、Pilkwang branch shortlist は exact archived source missing でブロックと判定した。
- `fle3n_final_blend` を `kentookumura/exp082-fle3n-final-source-infer` v1 として hidden-compatible source-port run し、commit output の submit-check / runtime / `/kaggle/input/notebooks` 非依存を確認した。
- `fle3n_final_blend` source-port v1 を ref `53885305` として提出し、Public LB `7.601` で ensemble route anchor を更新した。
