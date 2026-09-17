# Configs

再現可能な実験条件を YAML として保存する場所です。

```text
robot/       SO-101 の joint、actuator、制御周期
scene/       table、object、camera、初期配置
task/        instruction、phase、success 条件
dataset/     周波数、episode 長、保存 field、variation
policy/      π0.5 / SmolVLA の model と入出力
training/    step、batch size、freeze 範囲、seed
experiment/  上記 config の組み合わせと比較条件
```

## Rules

- 研究条件をコード内の隠れた default にしない
- 長さは m、時間は s、角度は rad を基本とし、例外は key 名へ単位を含める
- 乱数を使う処理には seed を記録する
- 未確定値を推測で埋めず、schema と実装を確認してから追加する
- 実験開始後に config を上書きせず、変更時は新しい実験 ID を作る

各 YAML の schema は次の開発段階で定義します。
