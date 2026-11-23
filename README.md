# T4 实训：译文质检

本项目批量计算 PPL、直接 NLL、条件 NLL 和 IFD，并用人工标签评估复查队列。仓库里的少量样本为合成演示数据。正式语料由项目负责人安排同事提供。

## 在 T4 上配置

```bash
cd /home/mineru_dev/projects/NLP_tutorial/NLP_tutorial
nvidia-smi
uv sync --locked --extra gpu
uv run --locked --extra gpu python -c 'import torch; print(torch.cuda.get_device_name())'
```

项目保留 Python 3.13。GPU 依赖声明在 `pyproject.toml` 的 `gpu` 可选依赖中，由 `uv.lock` 固定版本。依赖通过清华 TUNA PyPI 镜像安装到仓库的 `.venv`。Linux x86_64 上锁定的 PyTorch 2.8.0 使用 CUDA 12.8 构建。根据 `nvidia-smi` 的空闲显存，用 `CUDA_VISIBLE_DEVICES` 指定一张卡。例如在 GPU 4 空闲时使用 `CUDA_VISIBLE_DEVICES=4`。概率打分默认读取 NAS 上的 `/nas_data/models/Qwen/Qwen3.5-4B-Base`，对应仅预训练的 Base 版。开始前确认模型分片已下载完整。用其他模型做对照时，可通过 `--model /path/to/snapshot` 指定。T4 推理用 FP16，损失用 FP32。示例默认限定一次输入 512 token。

## 实验怎么进行

先用仓库自带的样本确认程序能跑通，再准备正式语料。正式实验按以下顺序进行：

1. 为同一段原文准备多份候选译文，人工核对并单独标注。按原始文档划出开发集和测试集。
2. 先跑开发集，用结果检查打分是否成功，并确定每种分数的排序方向、平局处理和人工复查预算。
3. 用冻结的配置跑测试集。各方法都按同一预算复查，例如只看各自排在最前的 20 条。
4. 汇总找到了多少真错和关键错误，再回看误报、漏报的原文和译文。

## 1. 先用演示样本试跑

`examples/` 提供两段原文，每段有三份候选，分别放在开发集和测试集。先在 T4 上用其中一张空闲卡运行下面这条命令。这里以 4 号卡为例。

```bash
CUDA_VISIBLE_DEVICES=4 uv run --locked --extra gpu python main.py score \
  --samples examples/samples.jsonl --probability \
  --mismatches examples/mismatches.jsonl --out runs/smoke/scores.jsonl
```

在输出里找 `b2`。这条译文把实测值 `0.08 mm` 写成了 `0.10 mm`。查看 `probability.ppl`、`probability.direct.nll`、`probability.conditional.nll` 和 `probability.conditional.ifd`，并核对 `probability.token_count` 和 `probability.token_check`。PPL 只看译文，流畅的错译也可能得到低分。实际语料到齐后，先抽约 10 条候选试跑，检查空文本、超长文本、分数缺失和原文位置是否能回查。

## 2. 准备正式语料和人工标签

### 选原文，给每段配三份译文

用项目负责人安排同事提供的材料，题材和文档类型以实际收到的语料为准。建议选约 **40 段原文，每段三份候选**，共约 120 条：

| 候选 | 准备方式 |
| --- | --- |
| 正确译文 | 由人工对照原文核对关键信息，保留一份可用译文。 |
| 流畅但译错的译文 | 选真实模型输出，或在正确译文里改动一处数字、单位、否定或术语。尽量保持句子自然。 |
| 有漏译、错配或表达问题的译文 | 优先选实际模型输出。样本不足时可人为删掉要点或换成同主题的错配译文。 |

保留一部分真实模型输出，也留一些含生僻术语或专名但译得正确的样本。这样可以检查方法会不会把难词一律当成错误。对含个人信息或其他敏感内容的材料先做脱敏，再准备候选和标签。

### 按原始文档划分开发集和测试集

先给每份原始文档一个 `doc_id`，再决定整份文档属于 `dev` 还是 `test`。同一文档的所有段落，以及每段原文的正确译文和错误变体，都留在同一侧。每段原文的候选共用 `group_id`，每个候选有独立的 `id`。

例如，约 40 段可以大致分成 10 段开发集、30 段测试集，具体数量以文档边界为准。开发集用于调排序和复查规则。测试集的人工标签留到最后核算成绩，避免根据测试结果反复调整规则。

### 保存输入和标签

用 UTF-8 JSONL，每行写一个 JSON 对象。`data/samples.jsonl` 保存候选；`data/labels.jsonl` 单独保存人工判断。两个文件的候选 ID 一一对应。

`data/samples.jsonl` 的一行：

```json
{"id":"b2","doc_id":"science-1","group_id":"b01","task":"translation","source":"实测值 0.08 mm，允许上限 0.10 mm。","candidate":"The measured value was 0.10 mm, with an upper limit of 0.10 mm.","location":{"line":1}}
```

`data/labels.jsonl` 中对应的一行：

```json
{"id":"b2","split":"dev","origin":"synthetic","error":true,"critical":true,"error_type":"number","evidence":"原文实测值 0.08 mm，候选译文实测值 0.10 mm"}
```

- `task` 固定写 `translation`。`source` 是原文，`candidate` 是实际要打分的译文。`location` 可填页码、段落或行号。语种默认中译英，其他语种在同组样本中一致写明 `source_language` 和 `target_language`。
- `origin` 用 `real` 标记实际获得的候选，用 `synthetic` 标记为实验构造或扰动的候选。
- 对照原文标注 `error`。关键事实、数值、单位、否定或重要术语的错译，以及关键内容漏译，可标为 `critical: true`。只影响通顺度的错误可标为 `error: true`、`critical: false`。正确译文的两个值都填 `false`。错误样本需写 `error_type` 和能回查的 `evidence`。
- 让同组同学先用几条样本统一标注口径。对有分歧的样本，回看原文后确定标签。

### 生成两个用于打分的输入文件

总表留给 `evaluate`。下列命令会检查文档有没有跨集合，并按标签中的 `split` 从总表生成 `data/samples.dev.jsonl` 和 `data/samples.test.jsonl`。写出的每行仍只有样本字段。

```bash
python3 - <<'PY'
from pathlib import Path

from nlp_tutorial.evaluate import read_labels
from nlp_tutorial.io import append_row, read_samples

data = Path("data")
samples = read_samples(data / "samples.jsonl")
labels = read_labels(samples, data / "labels.jsonl")

for split in ("dev", "test"):
    rows = [sample for key, sample in samples.items() if labels[key]["split"] == split]
    if not rows:
        raise SystemExit(f"{split} 没有样本")
    with (data / f"samples.{split}.jsonl").open("w", encoding="utf-8") as handle:
        for sample in rows:
            append_row(handle, sample)
PY
```

## 3. 用开发集确定规则，再跑测试集

先给开发集打分，再生成开发集报告和冻结配置。以下命令里 `--samples` 的两个用法不同：`score` 读取当前一侧的样本，`evaluate` 读取整个样本表、标签表和当前一侧的打分结果。

```bash
CUDA_VISIBLE_DEVICES=4 uv run --locked --extra gpu python main.py score \
  --samples data/samples.dev.jsonl --probability \
  --out runs/experiment/dev.scores.jsonl

uv run --locked --extra gpu python main.py evaluate \
  --samples data/samples.jsonl --labels data/labels.jsonl \
  --scores runs/experiment/dev.scores.jsonl --split dev --budget 20 \
  --settings runs/experiment/frozen.json --out runs/experiment/dev
```

先看开发集的 `summary.json`。脚本根据正确译文胜过错误译文的比例选择排序方向，再看关键错误和总命中数。确认模型版本、打分模板、预算和随机种子后，保留 `frozen.json`，用同一配置跑测试集。

```bash
CUDA_VISIBLE_DEVICES=4 uv run --locked --extra gpu python main.py score \
  --samples data/samples.test.jsonl --probability \
  --out runs/experiment/test.scores.jsonl

uv run --locked --extra gpu python main.py evaluate \
  --samples data/samples.jsonl --labels data/labels.jsonl \
  --scores runs/experiment/test.scores.jsonl --split test \
  --settings runs/experiment/frozen.json --out runs/experiment/test
```

测试集会使用冻结的预算和排序规则。分数相同的样本按固定种子排，打分失败的样本排在队尾。若复查到了失败样本，也计入预算。同一输出文件已有结果时，给下一轮实验另建目录。只跑随机抽查对照时，省略 `--extra gpu` 和 `--probability` 即可。

### 加做原文错配对照

从同一集合挑一批译文，为每条找一段题材相近但内容不同的替换原文。将开发集和测试集的错配分别存成 JSONL，每行形如：

```json
{"id":"b1","source":"试样厚度 0.60 mm，最低要求 0.40 mm。"}
```

在相应的 `score` 命令中加 `--mismatches data/mismatches.dev.jsonl` 或 `--mismatches data/mismatches.test.jsonl`。译文保持不变，所以直接 NLL 和 PPL 保持相同。比较匹配与错配原文下的条件 NLL 和 IFD，并检查两种方法的排序是否一致。`examples/mismatches.jsonl` 提供了可以直接试跑的例子。

## 4. 读结果并完成报告

| 输出 | 看什么 |
| --- | --- |
| `*.scores.jsonl` 和同名 `.meta.json` | 逐条 PPL、直接 NLL、条件 NLL、IFD、有效 token 数和失败原因；模型版本、模板、耗时及运行配置。PPL 与直接 NLL 共用一次推理，IFD 耗时为两次推理之和。 |
| `summary.json` | 对比随机抽查、PPL、条件 NLL 和 IFD。重点看前 20 条的真错与关键错误数、关键错误召回率、每找到一个真错需复查几条，以及同组正确译文的胜率和平局率。真实和合成结果分别看。 |
| `per_sample.csv` | 看每条样本的排名、是否进入复查、人工标签和原文位置。筛选 `false_alarm`（误报）和 `missed_error`（漏报），回看至少 3 条并解释原因。 |

报告应列出模型、样本划分、预算、打分成功率和耗时，并说明哪种分数适合筛查哪类错译。用人工标签核对至少一个样本的逐 token 结果。脚本计算的峰值显存是 PyTorch 在该 GPU 上已分配的峰值，包含模型。按任务书交付脱敏 JSONL、逐条结果和两页以内的报告。

## 请求 d3-flash

在 T4 上运行下面的命令，向内网 SystemOne 接口发一条分类请求。`state` 是要判断的内容，`questions` 里定义问题和选项。先用示例确认接口可用，再替换成同学们自己的验证问题。

```bash
curl -i --max-time 30 \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/d3-flash",
    "state": "Write a Python function that merges two sorted lists.",
    "questions": {
      "kind": {
        "type": "choice",
        "instructions": "What kind of work is this?",
        "criteria": {
          "code": "Writing or fixing code",
          "math": "Mathematics",
          "chat": "Anything else"
        }
      }
    }
  }' \
  http://192.168.0.213:18892/v1/systemone
```

响应中的 `answers.kind.choice` 是模型选中的选项。这个示例应返回 `code`。
