# T4 实训：OCR 正文与译文质检

本项目批量计算 PPL、直接 NLL、条件 NLL 和 IFD，提供 OCR 字符规则和结构化评审，并用人工标签评估复查队列。仓库里的少量样本为合成演示数据。正式语料由项目负责人安排同事提供。

## 在 T4 上配置

```bash
cd /home/mineru_dev/projects/NLP_tutorial/NLP_tutorial
nvidia-smi
uv sync --locked --extra gpu
uv run --locked --extra gpu python -c 'import torch; print(torch.cuda.get_device_name())'
```

项目保留 Python 3.13。依赖默认通过清华 TUNA PyPI 镜像安装。Linux x86_64 上的 PyTorch 通过上海交大镜像安装 CUDA 12.6 构建。根据 `nvidia-smi` 的空闲显存，用 `CUDA_VISIBLE_DEVICES` 指定一张卡。例如在 GPU 4 空闲时使用 `CUDA_VISIBLE_DEVICES=4`。模型默认是 `Qwen/Qwen2.5-1.5B`，首次运行会下载。已有权重可用 `--model /path/to/snapshot`，从 Hugging Face 加载时可用 `--revision <commit>` 固定版本。T4 推理用 FP16，损失用 FP32。示例默认限定一次输入 512 token。

## 跑通一批合成样本

```bash
CUDA_VISIBLE_DEVICES=4 uv run --locked --extra gpu python main.py score \
  --samples examples/samples.jsonl --probability \
  --mismatches examples/mismatches.jsonl --out runs/smoke/scores.jsonl

uv run --locked --extra gpu python main.py evaluate \
  --samples examples/samples.jsonl --labels examples/labels.jsonl \
  --scores runs/smoke/scores.jsonl --split dev --budget 1 \
  --settings runs/smoke/frozen.json --out runs/smoke/dev

uv run --locked --extra gpu python main.py evaluate \
  --samples examples/samples.jsonl --labels examples/labels.jsonl \
  --scores runs/smoke/scores.jsonl --split test \
  --settings runs/smoke/frozen.json --out runs/smoke/test
```

演示集很小，复查预算设为 1。正式实验默认预算为 20。只跑规则和随机基线时，安装和运行都可以省略 `--extra gpu`，评分时也省略 `--probability` 和 `--mismatches`。做概率打分时，可用 `--device cpu` 在 CPU 上排查问题。

### 添加 d3-flash 评审

在 T4 上，`--judge` 默认调用内网 SystemOne 接口 `http://192.168.0.213:18892/v1/systemone`，使用 `vllm-sr/d3-flash`。当前内网接口可直接访问。其他部署可用 `D3_BASE_URL` 覆盖 API 根地址或完整接口地址，用 `D3_MODEL` 覆盖模型名。服务需要认证时再设置 `D3_API_KEY`。

先用仓库里的合成样本试跑。只做评审时，`uv sync --locked` 即可安装所需依赖。`--no-sync` 会沿用现有环境。

```bash
uv run --locked --no-sync python main.py score \
  --samples examples/samples.jsonl --judge --repeat-first 1 \
  --out runs/systemone/scores.jsonl
```

接口用固定选项返回判项结果。对于判为失败的项目，脚本从输入切出候选证据，再用一次选择请求定位原文或译文短句。脚本逐字核对选中的证据。模型没有选出证据的判项记为 `insufficient_evidence`，并保留原始选项供复查。译文通顺度单独用候选译文评审。每次评审记录实测模型、请求次数、总请求耗时和可用的 token 用量。

`--repeat-first 4` 会对每组前 4 条样本各复跑一次。OCR 分为纯文本和附人工核对稿两种条件。服务只收到固定判项和相应的文本，评分命令不读标签。

## 正式实验的数据和步骤

- 样本 JSONL 每行至少含 `id`、`doc_id`、`task` 和 `candidate`。`task` 用 `ocr` 或 `translation`。OCR 的 `source` 是可选人工核对稿。译文另需 `group_id` 和 `source`，可填 `source_language` 和 `target_language`，默认中译英。`location` 用来回查。
- 人工标签单独成表，每行含 `id`、`split`（`dev` 或 `test`）、`origin`（`real` 或 `synthetic`）、布尔值 `error` 和 `critical`。有错误时填 `error_type` 和 `evidence`。同一原始文档必须落在同一侧，各译文候选共用一个 `group_id`。
- 先分别用每组 10 条样本试跑。正式实验可将样本按文档拆成两个输入文件。先计算开发集分数，再运行 `evaluate --split dev` 固定排序方向、最佳单项、组合策略和预算。冻结模板、模型版本和配置后计算测试集分数。两次评估的 `--samples` 和 `--labels` 都指向含两侧数据的总表，`--scores` 分别指向对应侧的分数文件和同目录的元数据。先在开发集查看失败率，确保各项方法已正常运行。
- 错配实验用 `--mismatches`，每行包含候选 `id` 和人工挑选的同主题替换原文 `source`。脚本保留同一译文的直接 NLL，分别比较两个原文下的条件 NLL 和 IFD。
- 两次概率打分都从候选的第二个 token 开始计分。结果记录 `token_count`、`token_check` 和两次 `first_token_nll`。检查第一条有效预测时，`previous_id` 对应 logits 行，`target_id` 是那个位置预测的下一个 token。

## 从结果到报告

`scores.jsonl` 保留逐条分数、证据、接口计时和失败原因。旁边的 `.meta.json` 记录版本、配置、加载耗时和整批运行耗时。峰值显存是 PyTorch 对单卡分配的峰值字节数，包含已加载的模型。PPL 和直接 NLL 共用一次推理，IFD 的耗时使用两次推理之和。

`summary.json` 给出固定预算下的命中、关键错误召回、每发现一错需复查几条、真实与合成数据明细、译文配对胜率和平局、评审失败和复跑一致性。`per_sample.csv` 可以按 `false_alarm` 或 `missed_error` 筛选误判，结合原文位置分析至少三个例子。报告用两页以内说明不同信号各自擅长的问题、成本，以及与开发集选出的最佳单项相比，组合多找出了多少关键错误。

相同分数按固定种子排序，未打分样本排到队尾且仍占复查预算。评审优先级为严重错误、轻微错误、证据不足、通过。纯 OCR 文本条件下，忠实度固定为证据不足，不用这项给整条样本提级。组合先按评审优先级排，再用 PPL 打破并列。开发集按译文配对胜率选概率分数方向，随后看关键错误和总命中数。最终最佳单项按开发集前 K 条中的关键错误和总命中数选出。
