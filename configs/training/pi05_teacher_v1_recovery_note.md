# π0.5 Teacher v1 checkpoint recovery

- Training steps: 30000
- Batch size: 16
- Total sample exposures: 480000
- Seed: 20260920
- Primary checkpoint: 030000_recovered
- Base checkpoint revision: 20459672e34d846d6c2b2d6bba1cbc8cecf12d38
- Environment: .venv-pi05
- torch.compile: enabled
- compile mode: default
- gradient checkpointing: enabled
- dtype: bfloat16
- freeze vision encoder: true
- train expert only: true

## Recovery

Training reached step 30000 and 480K sample exposures successfully.

During final checkpoint saving, the process terminated after writing the model
weights to a temporary safetensors file:

`030000/pretrained_model/.tmpVES7fQ`

The temporary file:

- had exactly the same byte size as the valid 025000 model.safetensors
- was successfully opened by safetensors
- contained 813 tensors

The preprocessor/postprocessor files were identical by SHA256 across
005000, 010000, 015000, 020000, and 025000.

Therefore, `030000_recovered/pretrained_model` was constructed from:

- 30000-step model weights from `.tmpVES7fQ`
- config.json from 030000
- frozen processor files from 025000

The recovered checkpoint was successfully loaded with `PI05Policy.from_pretrained`
and reported `All keys loaded successfully!`.

The original incomplete `030000` directory is preserved unchanged.
