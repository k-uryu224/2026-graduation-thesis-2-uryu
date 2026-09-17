# SO-101 VLA Local Policy Response Study

大分大学の卒業研究「研究テーマ(2)」のための実験リポジトリです。

MuJoCo 上の SO-101 を用い、同一のデータ・タスク・評価条件のもとで
π0.5 と SmolVLA を比較します。中心となる問いは、物体位置などの物理状態を
局所的に変化させたとき、両方の方策が行動をどのように変化させるかです。

現段階の目的は Response Distillation を先に実装することではありません。
まず、両モデルの **Local Policy Response（局所的な方策応答）** に再現可能な差が
存在するかを観測します。差が確認できた場合に限り、その応答を学習対象とする
次段階へ進みます。

## Research question

基準状態を \(s_0\)、局所的に変化させた状態を \(s_\delta\)、方策が出力する
行動を \(A(s)\) としたとき、次の応答を比較します。

\[
R(\delta) = A(s_\delta) - A(s_0)
\]

- 同じ物理変化に対して、π0.5 と SmolVLA はどのような行動変化を示すか
- 未学習位置でも、各方策の応答はタスク成功につながるか
- Compact VLA の弱点を、単純な成功率だけでなく局所的な応答として特定できるか

## Current pipeline

1. MuJoCo 上に SO-101・対象物・カメラを配置する
2. Script + Mink IK で再現可能な把持軌道を生成する
3. 共通の variation を用いて LeRobot Dataset を記録する
4. 同じデータ条件で π0.5 と SmolVLA を学習する
5. 学習位置と未学習位置でタスク成功率を評価する
6. 対応する状態間の action response を算出・比較する
7. 結果に基づき、Response 学習へ進むかを判断する

## Repository status

現在は **v0: repository scaffold + config schema draft** です。

- ディレクトリと責務の境界を定義済み
- Python パッケージとしての最小構成を定義済み
- 研究設計・座標系・データ形式・実験手順の記録場所を定義済み
- Phase 1 用の config YAML と最小の読み込み処理を定義済み
- MuJoCo、Mink、LeRobot、π0.5、SmolVLA の実装は未追加

未実装部分を動くように見せるダミーコードは置かず、確認できた機能から小さく
追加します。

## Repository structure

```text
.
├── configs/                  # 再現可能な実験条件
│   ├── robot/
│   ├── scene/
│   ├── task/
│   ├── dataset/
│   ├── policy/
│   ├── training/
│   └── experiment/
├── assets/                   # MuJoCo 用 robot / object assets
├── src/so101_vla_response/
│   ├── simulation/           # MuJoCo の load / reset / step
│   ├── ik/                   # Mink IK と target pose
│   ├── trajectory/           # 把持 phase と補間
│   ├── dataset/              # schema / recorder / variation
│   ├── policies/             # 共通 Policy と各モデル adapter
│   │   └── adapters/         # observation / action の共通表現変換
│   ├── response/             # response の計算・pair・schema
│   ├── training/             # 学習 loop と loss
│   ├── evaluation/           # success / action / response metrics
│   └── utils/
├── scripts/                  # 薄い CLI entry point
├── experiments/              # 実験目的・使用 config・結果要約
├── docs/                     # 研究設計と実験仕様
├── tests/                    # 小さく再現可能な自動テスト
├── data/                     # ローカルデータ（原則 Git 管理外）
└── results/                  # 生成結果（原則 Git 管理外）
```

## Design principles

- 研究条件は Python コードへ直書きせず `configs/` に保存する
- Simulation、IK、Trajectory、Dataset、Policy、Evaluation を直接結合しない
- π0.5 と SmolVLA は共通 Policy interface の背後へ置く
- Observation と Action を共通表現へ変換してから比較する
- 対応する実験条件には同じ `variation_id` を使い、physical pair を復元可能にする
- 実験結果には config、seed、commit hash、checkpoint を記録する
- Phase 1 ではモデル間の応答差を測り、結論を先取りして蒸留を実装しない

## Setup

現段階では開発用ツールだけを定義しています。MuJoCo などの研究依存関係は、
利用する SO-101 asset と各 API の互換性を確認した後に固定します。

### uv

```bash
git clone https://github.com/k-uryu224/2026-graduation-thesis-2-uryu.git
cd 2026-graduation-thesis-2-uryu
uv sync --extra dev
uv run pytest
```

### venv + pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## Documents

- [`docs/research_design.md`](docs/research_design.md): 研究範囲、段階、Go / No-Go
- [`docs/configuration.md`](docs/configuration.md): config の種類、単位、参照関係
- [`docs/coordinate_system.md`](docs/coordinate_system.md): world / robot / TCP / object 座標
- [`docs/dataset_format.md`](docs/dataset_format.md): episode、frame、variation、pair の仕様
- [`docs/experiment_protocol.md`](docs/experiment_protocol.md): データ生成から評価までの手順

## Development order

1. Config schema を定義する（進行中）
2. MuJoCo model をロードし、SO-101 の joint / site / camera を確認する
3. 座標系と `gripperframe` の意味を確定する
4. Mink IK で Pre-grasp / Grasp / Lift を到達可能にする
5. Script trajectory を生成する
6. LeRobot Dataset recorder を実装する
7. π0.5 / SmolVLA adapter を実装する
8. success と local policy response を評価する
