# 要件

## 依頼

`public_pf_beam_gate_only_audit` を実装する。PF/Beam を直接予測値や自由な残差補正として使わず、`base + w * (candidate - base)` の保守的な重み調整だけを train-side surrogate で監査する。

## 制約

- Route: `pf_beam`
- 親実験は `exp046_hidden_branch_surrogate_audit` とし、同じ `exp029` PF/Beam pseudo-test 生成物と split surface を使う。
- `exp026` anchor は audit split ごとに fold-safe に再生成する。
- learned candidate は `TVT` や残差を直接予測せず、bounded gate target のみを学習する。
- `w` は 0.2-0.4 程度の上限を持ち、全候補が `base + w * (candidate - base)` の形に収まる。
- 直接 PF residual / exp034-035 style meta residual の再投入はしない。
- audit-only とし、`submission.csv` は生成しない。

## 受け入れ基準

- `experiments/exp047_public_pf_beam_gate_only_audit/` に config、settings、train/inference notebook、監査スクリプト、notes がある。
- `config.yaml` に route、親、gate 候補、split、distance bucket、leakage policy が明記されている。
- train notebook は setup、入力確認、監査実行、metrics/生成物確認のセル構成になっている。
- 監査スクリプトは metrics、segment metrics、well metrics、diff metrics、gate stats、exp026 source summary を保存する。
- py_compile / ruff / validate-exp / small smoke が通る。
