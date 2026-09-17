# Experiments

実験の目的、条件、実行記録、要約を Git で残す場所です。大量の画像、動画、
checkpoint、frame 単位ログは `results/` に保存し、ここには置きません。

## Naming

```text
exp001_dataset_smoke_test/
exp002_pi05_baseline/
exp003_smolvla_baseline/
exp004_response_comparison/
```

## Minimum record

各実験の README に少なくとも次を記録します。

- Purpose
- Hypothesis or question
- Date
- Git commit
- Config paths
- Dataset ID / version
- Model checkpoint
- Random seeds
- Execution environment
- Metrics
- Result summary
- Failure notes
- Next decision

結果を見た後に元の config を書き換えず、新しい実験 ID として差分を残します。
