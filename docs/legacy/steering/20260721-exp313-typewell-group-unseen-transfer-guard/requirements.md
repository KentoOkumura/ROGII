# 要件

## 依頼

Type Well群の事前量をtestへ持つ前に、同群peerがない未知群・singleton・purged wellへの誤転送を防ぐ共通guardを設計する。今回はguard設計とscaffoldのみを作る。

## 制約

- Route: `pf_beam`。exp311/312の値は固定入力として扱う。
- availability/fallbackはouter-valid truthを見る前に決定する。
- testで使えるのはType Well content、visible prefix GR、raw geometryだけ。
- suffix TVT/error、formation train-only列、well ID ruleは禁止する。

## 受け入れ基準

- same-group well holdout、leave-one-group-out、spatial+typewell purgeを必須とする。
- exact groupはpeer wells≥2かつeffective rows≥64だけ利用する。
- default fallbackはidentity/no-correctionであり、parity誤差を1e-10以下にする。
- unseen groupは0.00 ft超のnegative transferを許容しない。
- このguardを通らない後続exp314〜320は実装しない。
