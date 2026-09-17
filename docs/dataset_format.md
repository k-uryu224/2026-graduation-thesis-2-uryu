# Dataset Format

## Purpose

Script + IK で生成する demonstration を LeRobot Dataset として記録し、通常学習、
方策比較、将来の physical pair 構築に共通利用します。

## Episode-level metadata

| Field | Meaning |
|---|---|
| `episode_id` | dataset 内で一意な episode ID |
| `variation_id` | 対応条件間で共有する variation ID |
| `task_id` | task の識別子 |
| `instruction` | 方策へ与える自然言語指示 |
| `seed` | variation 生成に使った seed |
| `object_baseline_pose` | 基準状態の object pose |
| `object_offset` | 基準状態からの displacement |
| `config_snapshot` | 生成に使用した config の参照または内容 |
| `success` | episode-level の成功判定 |

## Frame-level data

| Field | Meaning |
|---|---|
| `frame_index` | 0 始まりの frame index |
| `timestamp` | episode 開始からの時刻 [s] |
| `task_phase` | Home / Pre-grasp / Grasp / Close / Lift / Hold |
| `observation.images.*` | camera ごとの画像 |
| `observation.state` | 共通順序に正規化する前の robot state |
| `action` | 実行した robot action |
| `object_pose` | 必要に応じた simulator ground truth |

実際の LeRobot field 名・shape・dtype は、使用 version の schema を確認してから
固定します。上表は研究上失ってはいけない意味を示したものです。

## Variation invariants

同じ `variation_id` を異なる object position へ適用するとき、対象 perturbation 以外の
条件を一致させます。

- trajectory timing
- initial robot state
- camera parameters
- instruction
- noise sample または noise generation rule
- object physical properties

これにより、例えば次を physical pair として復元できます。

```text
variation_id = variation_007, object_offset = 0 mm
variation_id = variation_007, object_offset = +20 mm
```

## Response sample

Phase 2 で response dataset を作る場合も、元 episode との対応を失わない構造にします。

```text
pair_id
variation_id
base_episode_id
perturbed_episode_id
perturbation_type
perturbation_frame
perturbation_direction
perturbation_magnitude
base_action
perturbed_action
action_response
alignment_metadata
```

`action_response` だけを保存して元の state / action との関係を捨てないこと、また
success label や正解行動を observation へ混入させないことを原則とします。

## Validation requirements

- frame 数と timestamp が設定した周波数・時間に一致する
- camera、state、action の frame 数が一致する
- NaN / Inf がない
- joint と action の順序が metadata と一致する
- 同じ `variation_id` の pair で perturbation 以外の条件が一致する
- train / evaluation の位置と variation が意図せず重複しない
