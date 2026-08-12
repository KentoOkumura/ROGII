# 設計

## アプローチ

公開 `needless090/lb8-781-rogii-sel15-spread3` の PF/Beam primitive を実験内 module に移植し、submission ではなく train-side train well の途中以降を隠した疑似 test feature を生成する。

1. train well の horizontal log を読み、cutoff fraction で prefix と train well の途中以降を隠した疑似 test tail に分ける。
2. cutoff 以降の `TVT_input` を NaN にした frame を PF/Beam へ渡す。
3. scale 別 PF ensemble、14-config beam ensemble、公開 selector variant を計算する。
4. train well の途中以降を隠した疑似 test tail の各 row に対して prediction / confidence / bridge feature / diagnostic target を CSV に追記する。
5. well-level summary と metrics を保存し、後続の `public_sel15_pf_candidate_selector` / `public_sel15_pf_meta_stack` で読む。

## 実験範囲

- 対象実験: `exp029_public_sel15_pf_oof_feature_generation`
- Route: `pf_beam`
- 親実験: `exp027_public_replay_needless090_sel15_spread3`
- 変更する変数: 見えない test で使える train-side PF/Beam feature generation
- 固定する変数: 公開 replay anchor は変更しない。selector/meta model はこの実験では学習しない。

## リスク

- リークリスク: cutoff 以降の true `TVT` と `TVT_input` を PF/Beam に渡すと即リークになる。module 側で `TVT_input` を NaN に落とし、true `TVT` は出力診断列にのみ保持する。
- CV/LB 不一致リスク: 公開 sel15 route は Public LB には非常に強いが、train well の途中以降を隠した疑似 test で同じ強さが出る保証はない。この実験では提出判断をしない。
- ランタイム/メモリリスク: full 773 wells x 3 cutoffs x 128 seeds は重い。default は 20 wells / 1 cutoff / 16 seeds / 250 particles の smoke にし、full は override 後に Kaggle runtime を見て実行する。
