# Autoresearch v3

This directory is a Karpathy-style autonomous research setup for the chess corner model.

Design:
- `fixed_harness.py` is the immutable benchmark harness.
- `candidate.py` is the single mutable file the agent edits inside a run worktree.
- `run_commit.py` runs the current commit, logs a full experiment card, and recommends or applies `keep`/`promising`/`discard`.
- `init_run.py` creates a dedicated git worktree and initializes run metadata.
- `program.md` is the human-authored policy file for the agent.
- `EXPERIMENT_PLAN.md` is the current concrete architecture/testing plan.
- `templates/` contains starting points for the supported tracks.

The intent is to keep evaluation fixed while letting the agent iterate on the candidate model, preprocessing, augmentation, and optimizer logic.

Suggested flow:
1. Initialize a dedicated run worktree.
2. Open `program.md` in that worktree with Codex.
3. Let the agent modify only `candidate.py`.
4. Commit each experiment.
5. Run `run_commit.py --description "..."`
6. Keep, continue from `promising`, or discard based on the recorded metrics and guardrails.

Recommended tracks:
- `resnet_coords`: current coordinate-regression family
- `unet_dual_head`: segmentation + heatmaps + decoded coordinates

LLM controller:
- `llm_controller.py` drives a single autoresearch worktree from outside the worktree.
- It waits for any active benchmark in that worktree, uses the local `codex exec` CLI to update only `candidate.py`, commits, and launches the next `run_commit.py`.
- It uses your existing Codex/ChatGPT login. No separate OpenAI API key is required.
- Example:
  `python3 training/autoresearch_v3/llm_controller.py --worktree /tmp/codexchess_autoresearch/20260319-resnet_coords --model gpt-5.4 --reasoning-effort high`

One-command launcher:
- `start_loop.py` wraps the full setup for a track:
  - creates the worktree if needed
  - commits the initial template baseline so the controller starts clean
  - launches `llm_controller.py` with track-specific default budgets
- Prepare a fresh U-Net run without starting the loop yet:
  `python3 training/autoresearch_v3/start_loop.py --track unet_dual_head --prepare-only`
- Start the U-Net loop from your own terminal:
  `python3 training/autoresearch_v3/start_loop.py --track unet_dual_head | tee /tmp/codexchess_unet_loop.out`
- Reattach to an existing worktree later:
  `python3 training/autoresearch_v3/start_loop.py --track unet_dual_head --worktree /tmp/codexchess_autoresearch/20260324-unet_dual_head`

U-Net defaults:
- `start_loop.py` uses a longer initial budget for `unet_dual_head` (`3600s` budget, `300s` eval interval).
- The U-Net template now opts out of the legacy ResNet checkpoint fallback by default so the new track truly starts from scratch.
- If you want encoder warm-start anyway, pass `--resume training/autoresearch_gray_edges_mixed_models/best_corner_hybrid.pt`.
