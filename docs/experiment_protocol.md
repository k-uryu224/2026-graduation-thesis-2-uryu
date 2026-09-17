# Experiment Protocol

## 1. Reproducibility record

各実験を開始する前に次を固定・記録します。

- experiment ID
- Git commit
- config files
- Python、MuJoCo、Mink、LeRobot の version
- dataset ID と checksum
- model / checkpoint identifier
- random seeds
- hardware
- evaluation positions と試行数

## 2. Demonstration generation

MuJoCo asset を選定したら、まず環境ロードを確認します。

```bash
uv run python scripts/check_environment.py
```

このコマンドは `configs/experiment/response_comparison.yaml` から robot config を読み、
`robot.asset.model_xml` の MuJoCo model をロードして joint / site / camera 名を表示します。
STEP 4 で SO-101 asset を入れるまでは、`model_xml` 未設定として失敗するのが正常です。

1. scene と座標系を検証する
2. Home / Pre-grasp / Grasp / Close / Lift / Hold の target を設定する
3. Mink IK の到達性と joint limit を確認する
4. collision、grasp、lift、hold を viewer で確認する
5. variation set を seed から一度だけ生成する
6. 同じ variation set を各 training position へ適用する
7. 30 Hz、7 s、210 frames の設計どおり記録されることを検証する

## 3. Dataset validation

本学習前に小さい smoke-test dataset を作り、次を確認します。

- 画像・state・action を再生できる
- task phase と timestamp が単調かつ整合している
- action を再生して同じ動作を概ね再現できる
- object offset と variation metadata が正しい
- train / hold-out split に leakage がない

## 4. Policy training

π0.5 と SmolVLA は同じ demonstration と task instruction を使用します。モデル固有の
違いが必要な場合は消さずに記録し、公平性への影響を説明します。

- training step または sample exposure
- batch size と gradient accumulation
- optimizer / learning rate schedule
- trainable / frozen parameter 範囲
- checkpoint selection rule
- seed 数

評価結果を見て checkpoint を恣意的に選ばないよう、selection rule を事前に定めます。

## 5. Evaluation

少なくとも次を分離して集計します。

- task success
- grasp / lift / hold の失敗様式
- action error（定義した reference がある場合）
- local policy response
- training position と unseen position
- variation / seed ごとのばらつき

成功判定の高さ、保持時間、接触条件は simulator から安定して計測できることを確認後、
評価 config へ明記します。

## 6. Response comparison

1. 対応する episode / rollout pair を `variation_id` で抽出する
2. action を共通表現へ変換する
3. phase または明示した方法で時系列を alignment する
4. 基準状態との差として response を計算する
5. π0.5 と SmolVLA の方向・大きさ・時系列差を比較する
6. task success と failure mode との関係を調べる
7. 事前に定めた Go / No-Go 基準で次段階を判断する

## 7. Result preservation

生の出力は `results/`、再利用するデータは `data/`、実験の意味と要約は
`experiments/<experiment_id>/` に保存します。論文中の各表・図から、元の experiment
ID と集計コードまで追跡できる状態を維持します。
