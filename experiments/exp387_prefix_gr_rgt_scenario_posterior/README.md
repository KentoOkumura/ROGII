# exp387_prefix_gr_rgt_scenario_posterior

## 状態

- ルート: `pf_beam`
- 状態: exp386 Stage 0 FAILにより未実装で閉鎖
- CV: 未実行
- Public LB: 未提出
- 作成日: 2026-07-24
- 親実験: exp386
- 比較対象: exp226 / exp335 / exp385

## 仮説

exp386 が真の経路に近い複数の物理 scenario を事前に固定できるなら、既知 TVT prefix と観測 GR を outer-train の参照 GR template と照合することで、真値を使わずに scenario posterior を更新し、graph-cost prior の単純平均より正しい経路へ重みを寄せられる。

これは「3. 複数解＋GR尤度」のうち、exp386 の複数解生成に続く GR 尤度・事後分布部分に相当する。

## 設計

- exp386 の scenario 値・順序・参照 GR template を logical SHA で固定し、exp387 内では一切変更・再生成しない。
- target GR の robust level と first difference を、各 scenario の outer-train 参照 template と比較する。
- 尤度は固定した Student-t、窓長256行、stride 64、自由度4を用いる。
- exp386 graph cost を prior とし、共有 RGT node でのみ遷移できる exact forward-backward で posterior を計算する。
- 出力は scenario TVT の posterior mean とし、hard top-1 選択は禁止する。
- 実 GR と 512 行 circular-shift GR を比較し、尤度が地質対応を識別しているかを負例で確認する。

## 依存条件

次をすべて満たすまで実装・実行しない設計だった。

- exp386 Stage 0 PASS
- exp386 rolling-origin prefix PASS
- exp386 scenario oracle PASS
- exp386 scenario-bank manifest logical SHA の固定

## 検証方針

- Fold: exp386 と同じ outer 5-fold
- Group: `well_id`
- Stage 0: parent SHA、posterior 正規化、prefix heldout 改善、real-vs-circular 識別、leakage、resource gate
- Stage 1: pooled RMSE 7.20 ft以下、exp226 比 2.0 ft以上改善、5 fold 中4 fold以上で改善
- `1000+ ft` は 2.0 ft以上、hidden-like 2 scope は各1.5 ft以上改善
- `0–250 ft` の悪化は 0.05 ft以内
- Stage 2 の promotion safety は別途ユーザー承認を必要とする。

## 計算量

- scientific variant: 1
- Stage 0 likelihood audit: 1
- Stage 1 exact decoder: 773 wells
- LightGBM / HMM / PF / Beam: 0
- exp386 scenario 再生成: なし

## 実行入口

train / inference notebook はテンプレート scaffold のまま。exp386 の全 gate 合格、SHA 固定、別途実装承認までは変更・package・push・run しない。

## 結果

exp386 version 1はscenario-bank coverage `0.0`、cycle residual p95
`2.363303 > 0.10`でStage 0 FAIL_CLOSEとなった。固定scenario bankが生成されなかったため、
exp387は実装、Kaggle実行、CV、LBなしで閉じる。

## 所見

### リスク / 注意

- exp386 bank の oracle が弱ければ、GR 尤度だけでは LB 6.5 を狙えない。
- GR の層相変化が井戸間で再現しなければ、real-vs-circular 負例を分離できない。
- window、df、sigma、transition、temperature の事後 grid search と hard top-1 rescue は禁止する。
- 初回成功 run は deterministic anchor とせず、content SHA 一致 rerun を要求する。

### 次

現設計は再開しない。将来別のscenario generatorが独立した全gateとmanifest SHAを満たす場合は、
同じexp387を自動再開せず、新しい根拠と事前設計を確認する。
