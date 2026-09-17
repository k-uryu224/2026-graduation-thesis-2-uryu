# Research Design

## 1. Central objective

Strong VLA と Compact VLA を最終成功率だけで比較するのではなく、物体位置などの
物理状態を局所的に変化させたときの **行動の変化** を比較します。

初期段階では π0.5 を Teacher、SmolVLA を Student と固定して扱いません。
まず同一条件で両モデルを評価し、π0.5 の応答を学習対象とする妥当性を検証します。

## 2. Phase 1: response characterization

### Scope

- Simulator: MuJoCo
- Robot: SO-101
- Task: 単一対象物の top grasp
- Demonstration: Script + Mink IK
- Policies: π0.5、SmolVLA
- Main perturbation: 対象物位置の局所的な平行移動
- Main comparison: task success と local policy response

### Current experimental design

次の値は現在の研究設計です。実装に入れる前に、単位と座標軸を config schema 上で
明示します。

| Item | Current design |
|---|---|
| Recording frequency | 30 Hz |
| Episode duration | 7 s |
| Frames per episode | 210 |
| Training object offsets | -20 mm, 0 mm, +20 mm |
| Training variations | 20 per training position |
| Hold-out variations | 5 |
| Demonstration phases | Home, Pre-grasp, Grasp, Close, Lift, Hold |

評価位置、各モデルの学習条件、action の共通表現、response metric の集計方法は、
実装と予備実験を通して確定します。未確定値は結果を見ながら都合よく変更せず、
変更理由と実験 ID を残します。

## 3. Local policy response

基準状態 \(s_0\) と変化後の状態 \(s_\delta\) に対して、同一の時刻・task phase・
variation に対応する行動を共通 action space へ変換し、次を計算します。

\[
R_\pi(\delta) = A_\pi(s_\delta) - A_\pi(s_0)
\]

比較対象には少なくとも次を含めます。

- response の方向
- response の大きさ
- 時系列上の response
- task phase ごとの response
- response と task success の関係

異なる長さの rollout を直接 frame index だけで対応付ける方法は採用せず、phase、
timestamp、または alignment 手法を明記します。

## 4. Go / No-Go

Phase 2 へ進む前に、少なくとも次を確認します。

- 物理 perturbation に対する π0.5 の応答が再現可能である
- SmolVLA との差が複数 seed / variation で観測できる
- 差が action 表現や時間 alignment の人工的な差ではない
- 応答差が task performance または失敗様式と関係する

判定基準の数値は Phase 1 の評価 protocol を確定するときに事前登録します。

## 5. Phase 2: response learning candidate

Go の場合のみ、既存構造へ次を追加します。

- physical pair dataset
- teacher response generation
- point-wise knowledge distillation baseline
- response loss
- shuffled-pair control
- response weight \(\lambda\) の比較
- 未学習位置・異なる perturbation magnitude での汎化評価

Phase 2 は Phase 1 のコードを置き換えず、`response/`、`training/`、`scripts/` へ
追加して構成します。
