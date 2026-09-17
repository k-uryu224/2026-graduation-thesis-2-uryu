# Scripts

実験を開始するための薄い CLI entry point を置きます。

Script は config の読み込みと `src/so101_vla_response/` の呼び出しだけを担当し、
MuJoCo、IK、学習、評価の本体を直接実装しません。

予定している entry point:

- `check_environment.py`
- `generate_demo.py`
- `generate_dataset.py`
- `train_policy.py`
- `run_policy.py`
- `evaluate.py`
