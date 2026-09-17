# Coordinate System

## Purpose

物体 perturbation、IK target、action response を同じ意味で扱うための座標系仕様です。
MuJoCo viewer と model 定義を確認するまでは、world の x / y の向きを推測で確定しません。

## Canonical frames

| Name | Owner | Intended use | Status |
|---|---|---|---|
| `world` | MuJoCo | scene 全体の基準 | 軸の向きを要確認 |
| `robot_base` | SO-101 asset | robot-relative target | body / site 名を要確認 |
| `gripperframe` | SO-101 asset | end-effector / TCP candidate | 原点・姿勢を要確認 |
| `object_center` | object asset | object pose と perturbation | geometry 定義後に確定 |
| `camera_*` | scene | visual observation | extrinsic を要記録 |

## Required conventions

- Position: m
- Orientation: quaternion の並びを API ごとに明記する
- Joint angle / command: rad または正規化値を field 名と adapter で区別する
- Timestamp: episode 開始からの s
- Object offset: 基準位置からの signed displacement

## Values to verify before implementation

| Item | Value | Verification |
|---|---|---|
| World up axis | TBD | MuJoCo model / viewer |
| Positive perturbation axis | TBD | marker を置いて viewer で確認 |
| Table top height | TBD | geom pose + size から算出 |
| Object center at baseline | TBD | scene config と viewer |
| `gripperframe` origin | TBD | site marker と mesh の比較 |
| `gripperframe` approach axis | TBD | local-axis visualization |
| Camera extrinsics | TBD | model と rendered image |

## Verification procedure

1. SO-101 asset を単独でロードする
2. world origin と xyz marker を表示する
3. robot base frame を表示する
4. `gripperframe` に小さな marker を置く
5. 対象物を各軸へ既知量だけ移動し、符号と単位を確認する
6. camera 名、位置、向き、画像の上下左右を記録する
7. 確認済み値を config とこの文書へ同時に反映する

座標の確認が完了するまで、`x_perturb` のような曖昧な名前を研究結果へ使用しません。
確定後は `world_x_offset_m` など、frame と単位が分かる名前を使用します。
