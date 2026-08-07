# exp069_pixiux_pf_beam_direct_submit_audit

## 状態

- ルート: pf_beam
- 状態: completed
- CV: -
- Public LB: 9.721
- Private LB: -
- Submit ID: 53706005
- 作成日: 2026-06-13
- 親実験: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`

## 仮説

exp063 は公開 Pixiux notebook の LightGBM replay `lgb_mean` を提出して Public LB 8.811 だった。LightGBM を通さず、同じ Pixiux likelihood-PF / PF-Beam 予測値を直接提出すると、LightGBM が拾った誤差または過補正を避けられる可能性がある。

## 変更点

- exp063 の公開 replay feature generation を再利用。
- LightGBM booster loading / prediction を使わない。
- 既定提出候補は raw `likpf_mean`。
- `likpf_scale_*`、`pf_ancc`、`pf_z`、Beam 系 direct candidates は診断用に保存する。

## 検証方針

- Fold: なし。直接提出監査。
- Group: なし。test-side feature generation のみ。
- Stratification: なし。
- Leakage Check: train target、OOF prediction、static visible override、hidden-specific branch、final public blend、projection postprocess を使わない。

## 実行入口

- 学習 notebook: `exp069_pixiux_pf_beam_direct_submit_audit_train.ipynb`
- 推論 notebook: `exp069_pixiux_pf_beam_direct_submit_audit_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp069_pixiux_pf_beam_direct_submit_audit`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | 9.721 |
| Private LB | - |
| Kaggle inference | v3 deterministic complete |
| Submit-check | PASS |

## 所見

### 良かった点

- CPU-only Kaggle inference v3 が完了し、`submission.csv` は submit-check PASS。
- `likpf_mean` は exp027 / exp063 と高相関だが同一ではなく、Public LB を見る価値のある別候補になった。
- deterministic v3 code submission は Public LB 9.721 で、pre-patch v2 の 9.877 よりは改善したが、exp063 v2/best 8.811 より +0.910、exp027 8.781 より +0.940 悪化した。

### 悪かった点

- CV は作らないため、提出後の Public LB と exp063 / exp027 との差分で判断する。
- pre-patch v2 は Public LB 9.877、deterministic v3 は 9.721 で、どちらも exp027 / exp063 v2/best より悪い。v2 と v3、また他提出 ref を混ぜない。

### リスク / 注意

- hidden test では test rows が増えるため、PF / likelihood-PF feature generation runtime が支配的になる。
- `ref=53710264` / `ref=53710105` の Public LB 8.766 は exp069 への紐づけ誤りとして扱う。

## 次

- 採用しない。今後は direct submission ではなく、PF/Beam disagreement、confidence、error map の診断値として使う。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
