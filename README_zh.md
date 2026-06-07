# ReAct Orchestrator 中文运行说明

`orchestrator/` 是本项目的 Agent 总控层。它不直接修改 `Computer-Graphics/` 的代码，而是把 `Computer-Graphics/` 里的脚本和接口包装成 Agent 可调用的工具，并把每一步写入 `agent_trace.json`。

当前默认流程是规则型 ReAct：

```text
Thought -> Action -> Observation -> Verification -> Decision
```

默认不会自动跑 SAM3 / YOLO 这类重模型，也不会自动跑耗时的最终合成；这些工具已经注册好，可以单独调用。

## 1. 目录说明

```text
orchestrator/
  configs/
    demo_cases.json        # demo case 配置
    tool_registry.json     # Agent 工具注册表
  scripts/
    run_react_demo.py      # 默认 ReAct demo
    run_tool.py            # 单独调用某个工具
    run_llm_smoke.py       # LLM 连通性测试
  src/orchestrator/
    agent.py               # ReAct 主流程
    llm_client.py          # OpenAI-compatible LLM client
    tools/                 # 阶段 1 / 阶段 3 / 插入候选工具
    adapters/              # 当前接口 adapters
    verifiers/             # verifier
    policies/              # 决策策略
  outputs/
    g1_p1/
      agent_trace.json
```

## 2. 推荐运行环境

本机推荐使用 `check-numpy` conda 环境运行真实候选生成和阶段 3 光照工具：

```bash
conda run -n check-numpy python --version
```

如果环境缺少 Pillow，先安装：

```bash
conda run -n check-numpy python -m pip install Pillow
```

只跑 dry-run 时可以直接用系统 Python：

```bash
python3 orchestrator/scripts/run_react_demo.py --dry-run
```

## 3. 运行默认 ReAct Demo

### 3.1 Dry-run

不调用 `PersonInserter`，只检查流程和 trace：

```bash
python3 orchestrator/scripts/run_react_demo.py --dry-run
```

预期输出类似：

```text
case=g1_p1 final_status=success_dry_run
trace=.../orchestrator/outputs/g1_p1/agent_trace.json
#1 vision.inspect_existing_artifacts: ... -> pass
#2 insertion.find_candidates: Dry run ... -> skipped
```

### 3.2 真实生成插入候选

调用当前 `Computer-Graphics/PersonInserter.py` 的 `find_insertion_patches(...)`：

```bash
conda run -n check-numpy python orchestrator/scripts/run_react_demo.py
```

预期输出类似：

```text
case=g1_p1 final_status=candidates_ready
#1 vision.inspect_existing_artifacts: group persons=22, person records=1, group masks=23, person masks=1 -> pass
#2 insertion.find_candidates: Found 5 insertion candidate(s). -> pass
```

生成文件：

```text
orchestrator/outputs/g1_p1/agent_trace.json
orchestrator/outputs/g1_p1/insertion/candidate_summaries.json
```

## 4. 配置 Demo Case

默认 case 在：

```text
orchestrator/configs/demo_cases.json
```

当前默认是：

```json
{
  "case_id": "g1_p1",
  "goal": "insert_person_into_group_photo",
  "computer_graphics_root": "Computer-Graphics",
  "group_id": "g1",
  "person_id": "p1",
  "top_k": 5
}
```

如需切换到其他组合，可以改 `group_id` 和 `person_id`，例如 `g2` + `p2`。

## 5. 单独调用 Agent 工具

工具入口：

```bash
python3 orchestrator/scripts/run_tool.py <tool_name>
```

可用工具见：

```text
orchestrator/configs/tool_registry.json
```

### 5.1 检查已有感知产物

```bash
python3 orchestrator/scripts/run_tool.py vision.inspect_existing_artifacts
```

### 5.2 Dry-run SAM3 mask 生成

不会真实调用模型，只展示将要执行的命令：

```bash
python3 orchestrator/scripts/run_tool.py vision.generate_sam3_masks --dry-run \
  --params-json '{"image_selection":"group","checkpoint":"/path/to/sam3.pt"}'
```

参数说明：

- `image_selection`: `group` / `person` / `both` / `all_material`
- `checkpoint`: SAM3 checkpoint 路径
- `confidence`: 置信度阈值
- `device`: `cuda` 或 `cpu`
- `max_inference_side`: 推理最长边

### 5.3 Dry-run YOLO mask 生成

```bash
python3 orchestrator/scripts/run_tool.py vision.generate_yolo_person_masks --dry-run
```

可传参数示例：

```bash
python3 orchestrator/scripts/run_tool.py vision.generate_yolo_person_masks --dry-run \
  --params-json '{"image_selection":"both","model":"models/yolo/yolo11l-seg.pt","conf":0.35}'
```

### 5.4 从 mask 提取 metadata

```bash
python3 orchestrator/scripts/run_tool.py vision.extract_metadata_from_masks --dry-run
```

如需指定 mask 目录：

```bash
python3 orchestrator/scripts/run_tool.py vision.extract_metadata_from_masks --dry-run \
  --params-json '{"masks_dir":"sam3_masks","contour_stride":1}'
```

### 5.5 阶段 3 光照一致性 smoke test

调用 `ImageCompositor.MRFImageCompositor` 在一个小 synthetic patch 上测试：

```bash
conda run -n check-numpy python orchestrator/scripts/run_tool.py compositing.run_light_smoke \
  --params-json '{"size":24,"max_iter":5}'
```

输出：

```text
orchestrator/outputs/g1_p1/final/light_smoke.png
orchestrator/outputs/g1_p1/final/light_smoke_report.json
```

### 5.6 阶段 3 真实合成 top candidate

该步骤会比较耗时，因为会调用 MRF 合成：

```bash
conda run -n check-numpy python orchestrator/scripts/run_tool.py compositing.compose_top_candidate \
  --params-json '{"candidate_rank":1,"max_iter":200,"max_crop_size":200}'
```

如果只想检查计划，不真实合成：

```bash
conda run -n check-numpy python orchestrator/scripts/run_tool.py compositing.compose_top_candidate --dry-run
```

## 6. LLM 配置与运行

Orchestrator 已支持 OpenAI-compatible LLM 调用。

默认配置：

```text
base_url=https://api.siliconflow.cn/v1
model=nex-agi/Nex-N2-Pro
```

### 6.1 配置 API Key

在项目根目录 `.env` 或 `orchestrator/.env` 中写入：

```text
API_KEY=你的 SiliconFlow API Key
```

也支持：

```text
SILICONFLOW_API_KEY=你的 SiliconFlow API Key
```

注意：不要把 `.env` 提交到公开仓库。

### 6.2 LLM dry-run

只检查配置，不发起网络请求：

```bash
python3 orchestrator/scripts/run_llm_smoke.py --dry-run
```

预期输出：

```json
{
  "status": "dry_run_ok",
  "base_url": "https://api.siliconflow.cn/v1",
  "model": "nex-agi/Nex-N2-Pro",
  "env_loaded": true
}
```

### 6.3 真实 LLM smoke test

会调用 SiliconFlow API：

```bash
python3 orchestrator/scripts/run_llm_smoke.py
```

预期输出类似：

```json
{
  "status": "success",
  "base_url": "https://api.siliconflow.cn/v1",
  "model": "nex-agi/Nex-N2-Pro",
  "content": "{\"status\":\"ok\"}"
}
```

### 6.4 自定义 prompt

```bash
python3 orchestrator/scripts/run_llm_smoke.py \
  --prompt "请用 JSON 返回 {status: ok, message: 简短说明}"
```

## 7. 当前 Agent 的能力边界

当前默认 ReAct demo 仍然是规则型流程：

```text
检查已有感知产物 -> 生成插入候选 -> verifier -> trace
```

LLM client 已经接入，但默认主流程还没有让 LLM 自动决定下一步。后续可以把 LLM 接入 `llm_decision_policy.py`，让它在白名单工具中选择下一步，例如：

```text
vision.generate_sam3_masks
vision.generate_yolo_person_masks
vision.extract_metadata_from_masks
insertion.find_candidates
compositing.compose_top_candidate
```

这样可以从固定流程升级为 LLM-assisted ReAct Agent。

## 8. 常见问题

### 8.1 `No module named 'PIL'`

在 `check-numpy` 环境安装 Pillow：

```bash
conda run -n check-numpy python -m pip install Pillow
```

### 8.2 NumPy / SciPy ABI 报错

如果系统 Python 出现类似：

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

请使用 `check-numpy` conda 环境运行真实工具：

```bash
conda run -n check-numpy python orchestrator/scripts/run_react_demo.py
```

### 8.3 SAM3 checkpoint 缺失

`vision.generate_sam3_masks` 需要 checkpoint 或允许 Hugging Face 下载。建议先 dry-run 检查命令：

```bash
python3 orchestrator/scripts/run_tool.py vision.generate_sam3_masks --dry-run \
  --params-json '{"checkpoint":"/path/to/sam3.pt"}'
```

### 8.4 只想看 trace

运行：

```bash
python3 orchestrator/scripts/run_react_demo.py --dry-run
```

然后查看：

```text
orchestrator/outputs/g1_p1/agent_trace.json
```

## 9. LLM 决策接入 ReAct Demo

现在 `run_react_demo.py` 支持在默认规则流程结束后，让 LLM 读取当前 trace 和 verifier 结果，并从白名单工具里选择下一步。

默认行为：

```text
规则流程：inspect_existing_artifacts -> find_insertion_candidates
LLM 决策：读取 trace/verifier -> 选择 stop 或一个白名单工具
```

### 9.1 只让 LLM 做决策，不执行它选中的工具

这是推荐的安全测试方式。它会真实请求 LLM，但不会自动跑耗时工具：

```bash
conda run -n check-numpy python orchestrator/scripts/run_react_demo.py --use-llm-decision
```

预期输出类似：

```text
case=g1_p1 final_status=llm_decision_stop
#1 vision.inspect_existing_artifacts: ... -> pass
#2 insertion.find_candidates: Found 5 insertion candidate(s). -> pass
#3 llm.choose_next_action: LLM selected stop: ... -> pass
```

对应 trace 会多出一步：

```json
{
  "action": {
    "tool": "llm.choose_next_action"
  },
  "observation": {
    "summary": "LLM selected stop: ..."
  }
}
```

### 9.2 指定 LLM 可选工具白名单

默认白名单包括：

```text
compositing.run_light_smoke
compositing.compose_top_candidate
vision.extract_metadata_from_masks
vision.inspect_existing_artifacts
```

可以手动覆盖：

```bash
conda run -n check-numpy python orchestrator/scripts/run_react_demo.py \
  --use-llm-decision \
  --llm-allowed-tools compositing.run_light_smoke,vision.inspect_existing_artifacts
```

LLM 只能从白名单里选，或者选择 `stop`。如果它返回白名单之外的工具，policy 会把决策改为 `stop`。

### 9.3 允许执行 LLM 选中的工具

默认只记录 LLM 决策，不执行。若要执行 LLM 选中的工具，加：

```bash
conda run -n check-numpy python orchestrator/scripts/run_react_demo.py \
  --use-llm-decision \
  --execute-llm-tool
```

注意：如果 LLM 选择 `compositing.compose_top_candidate`，会运行 MRF 合成，可能比较耗时。建议第一次测试时把白名单限制到轻量工具：

```bash
conda run -n check-numpy python orchestrator/scripts/run_react_demo.py \
  --use-llm-decision \
  --execute-llm-tool \
  --llm-allowed-tools compositing.run_light_smoke
```

### 9.4 指定 LLM endpoint 或模型

默认：

```text
base_url=https://api.siliconflow.cn/v1
model=nex-agi/Nex-N2-Pro
```

如需覆盖：

```bash
conda run -n check-numpy python orchestrator/scripts/run_react_demo.py \
  --use-llm-decision \
  --llm-base-url https://api.siliconflow.cn/v1 \
  --llm-model nex-agi/Nex-N2-Pro
```

### 9.5 当前 LLM 决策策略

LLM 决策由以下文件实现：

```text
orchestrator/src/orchestrator/policies/llm_decision_policy.py
```

它会把以下信息发给 LLM：

- `case_id`
- `goal`
- 最近几步 ReAct trace
- 最新 verifier 结果
- 允许选择的工具白名单

要求 LLM 返回：

```json
{
  "next": "tool_name_or_stop",
  "reason": "选择原因",
  "params": {}
}
```

当前 system prompt 会要求 LLM：

- 只能从白名单选择工具。
- 不要编造工具。
- 如果已有候选并且没有必要继续，就选择 `stop`。
- 优先选择安全、低成本诊断工具。
