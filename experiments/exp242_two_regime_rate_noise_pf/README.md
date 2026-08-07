# exp242 two-regime rate-noise PF

exp072-compatible likelihood-PFの粒子へ`smooth / turn`の固定2状態を追加し、`turn`時だけ
rate process noiseを4倍にするtrain-side監査です。観測尤度、position noise、particle/seed数、
resamplingは変更しません。

## 状態

- Route: `pf_beam`
- 状態: Kaggle train v2完了・不採用
- 新規variant: `two_regime_k4` 1件
- LightGBM config / fold / booster: `0 / 0 / 0`
- inference / submission: guard不通過のため実施しない

## 仮説

通常の滑らかなrate遷移を保ったまま、約1%の`turn`粒子だけに広いrate process noiseを
許すと、全粒子を高noise化せずに急なrate変化を追跡できる。

## 検証方針

exp072 pseudo-tailの全eligible wellsで`two_regime_k4`だけを生成し、保存済みexp072
`likpf_mean`とoverall、距離帯、`1000_plus`、hidden-like、worst-wellを比較します。
regime occupancy、posterior mass、switch、ESS、resamplingも診断します。

## 所見

3,783,989 rows / 773 wellsで完走しましたが、RMSEはexp072 `likpf_mean`の11.594898から
13.254455へ+1.659557悪化しました。全distance bucketとhidden-like 2群も悪化し、
turn posterior mass 0.017897はparticle fraction 0.018088を上回りませんでした。

## 次のアクション

追加grid、raw-test inference、submissionには進みません。後続ではdynamic high-noise regimeではなく、
known prefixだけから作る離散的な初期rate候補をbaseと並存させる方向を優先します。

## 固定設定

- transition matrix: `[[0.9998, 0.0002], [0.02, 0.98]]`
- 初期粒子: `smooth=495`, `turn=5`
- smooth rate noise: `0.002`
- turn rate noise: `0.008`
- particles / seeds: `500 / 128`

詳細は`config.yaml`、実行記録は`SESSION_NOTES.md`を参照してください。
