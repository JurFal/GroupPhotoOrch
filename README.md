# ReAct Orchestrator

This folder is the non-invasive Agent layer for the Computer-Graphics project.
It lives at `Computer-Graphics/orchestrator/`, treats `Computer-Graphics/` as
the tool workspace, and writes all Agent traces
under `orchestrator/outputs/`.

P0 scope:

- Do not modify `Computer-Graphics/` during Agent execution.
- Read the current `person_metadata/` and `sam3_masks/` artifacts directly.
- Call `PersonInserter.find_insertion_patches` through an adapter.
- Record ReAct steps in `agent_trace.json`.
- Verify raw vision artifacts, candidate summaries, HSV tone, and final MRF lighting output.

Startup snippets below enter `Computer-Graphics/` first; omit that line if you
are already there.

Run a dry run:

```bash
cd Computer-Graphics
python orchestrator/scripts/run_react_demo.py --dry-run
```

Run the full deterministic flow against existing metadata. On this machine, use the `check-numpy` conda environment:

```bash
cd Computer-Graphics
conda run -n check-numpy python orchestrator/scripts/run_react_demo.py
```

The default flow now continues past HSV tone alignment: a lighting verifier
selects a reduced MRF iteration budget, then `compositing.compose_top_candidate`
runs the final MRF pass. HSV alone is treated as a pre-adjustment, not the final
output.

If the environment is missing image dependencies, install Pillow into it first:

```bash
cd Computer-Graphics
conda run -n check-numpy python -m pip install Pillow
```

## Calling Individual Tools

The Agent toolbox can call registered tools directly:

```bash
cd Computer-Graphics
python orchestrator/scripts/run_tool.py vision.generate_sam3_masks --dry-run
python orchestrator/scripts/run_tool.py vision.generate_yolo_person_masks --dry-run
python orchestrator/scripts/run_tool.py vision.extract_metadata_from_masks --dry-run
conda run -n check-numpy python orchestrator/scripts/run_tool.py geometry.find_insertion_candidates
conda run -n check-numpy python orchestrator/scripts/run_tool.py compositing.align_tone_hsv
conda run -n check-numpy python orchestrator/scripts/run_tool.py compositing.run_light_smoke
conda run -n check-numpy python orchestrator/scripts/run_tool.py compositing.compose_top_candidate \
  --params-json '{"candidate_rank":1,"max_iter":24,"max_crop_size":140,"use_active_solve":true}'
```

Stage split:

- `stage2_tools.py`: geometry and insertion planning through `PersonInserter.find_insertion_patches`.
- `stage3_tools.py`: HSV tone alignment, MRF light smoke, and final MRF compositing.
- `lighting_verifier.py`: always plans a final MRF pass after HSV and chooses `max_iter`/`max_crop_size` from tone mismatch and candidate size.
- `insertion.find_candidates` remains as a backward-compatible alias for `geometry.find_insertion_candidates`.

Stage 2 writes lightweight summaries plus cached patch artifacts under
`orchestrator/outputs/<case>/insertion/patches/`. Stage 3 prefers those cached
patches, so final MRF compositing does not rerun geometry planning. The base
MRF solver also uses a local active-mask sparse matrix by default
(`use_active_solve=true`) to avoid solving over the whole crop background.

Tool-specific parameters can be passed with `--params-json`:

```bash
cd Computer-Graphics
python orchestrator/scripts/run_tool.py vision.generate_sam3_masks --dry-run \
  --params-json '{"image_selection":"group","checkpoint":"/path/to/sam3.pt"}'
```

## Optional LLM Client

The orchestrator includes an OpenAI-compatible chat client for SiliconFlow.
It defaults to:

```text
base_url=https://api.siliconflow.cn/v1
model=nex-agi/Nex-N2-Pro
```

Set the key in the project root `.env` or `orchestrator/.env`:

```text
API_KEY=your_siliconflow_key
```

`SILICONFLOW_API_KEY` is also accepted.

Dry-run config check:

```bash
cd Computer-Graphics
python orchestrator/scripts/run_llm_smoke.py --dry-run
```

Real smoke test:

```bash
cd Computer-Graphics
python orchestrator/scripts/run_llm_smoke.py
```

The default ReAct demo still uses deterministic rules. The LLM client is
available for future decision policies and can choose only from whitelisted
registry tools when used through `llm_decision_policy.py`.
