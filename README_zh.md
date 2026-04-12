<p align="center">
  <h1 align="center">SocialSciKit</h1>
  <p align="center">
    <strong>面向社会科学研究者的零代码文本分析工具包</strong>
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-562%20passing-brightgreen.svg" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="#"><img src="https://img.shields.io/badge/UI-Gradio-orange.svg" alt="Gradio UI"></a>
  <a href="#"><img src="https://img.shields.io/badge/i18n-EN%20%7C%20%E4%B8%AD%E6%96%87-blueviolet.svg" alt="i18n"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>中文</strong>
</p>

---

## 简介

SocialSciKit 是一个开源 Python 工具包，让社会科学研究者**无需编写任何代码**即可完成文本分析。基于 Gradio 构建的 Web 界面，完整支持中英双语。

两大核心模块：

- **QuantiKit** — 文本分类全流程（方法推荐 → 标注 → 提示词/微调分类 → 评估 → 导出）
- **QualiKit** — 质性编码全流程（上传 → 脱敏 → 研究框架 → LLM 编码 → 人工审核 → 导出）

---

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [QuantiKit：文本分类](#quantikit文本分类)
- [QualiKit：质性编码](#qualikit质性编码)
- [支持的 LLM 后端](#支持的-llm-后端)
- [示例数据集](#示例数据集)
- [项目结构](#项目结构)
- [关键文献](#关键文献)
- [引用](#引用)
- [开发指南](#开发指南)
- [许可证与免责声明](#许可证与免责声明)
- [作者](#作者)

---

## 安装

### 环境要求

- Python 3.9 或更高版本
- pip（Python 包管理器）

### 方式 A：从 PyPI 安装

```bash
pip install socialscikit
```

### 方式 B：从源码安装

```bash
git clone https://github.com/Baron-Sun/socialscikit.git
cd socialscikit
pip install -e .
```

### 核心依赖

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `gradio` | &ge; 4.0 | Web UI 框架 |
| `pandas` | &ge; 2.0 | 数据处理 |
| `openpyxl` | 任意 | Excel 读写 |
| `spacy` | &ge; 3.7 | NLP 管线（分词、命名实体识别） |
| `transformers` | &ge; 4.40 | 模型微调（RoBERTa / XLM-R） |
| `datasets` | 任意 | HuggingFace 数据集 |
| `openai` | &ge; 1.0 | OpenAI API 客户端 |
| `anthropic` | &ge; 0.25 | Anthropic API 客户端 |
| `scikit-learn` | 任意 | 评估指标计算 |
| `scipy` | 任意 | 统计计算 |
| `bertopic` | 任意 | 主题建模 |
| `presidio-analyzer` | 任意 | PII 检测引擎 |
| `presidio-anonymizer` | 任意 | PII 匿名化 |
| `langdetect` | 任意 | 语言检测 |
| `tiktoken` | 任意 | Token 计数 |
| `httpx` | 任意 | Ollama HTTP 客户端 |
| `rich` | 任意 | CLI 格式化输出 |

### 可选：spaCy 语言模型

为获得最佳脱敏效果，建议下载至少一个 spaCy 模型：

```bash
# 英文
python -m spacy download en_core_web_sm

# 中文
python -m spacy download zh_core_web_sm
```

---

## 快速开始

### 启动统一界面（推荐）

```bash
socialscikit launch
# 或直接：
socialscikit
# 浏览器打开 http://127.0.0.1:7860
```

### 单独启动模块

```bash
# 仅 QuantiKit
socialscikit quantikit --port 7860

# 仅 QualiKit
socialscikit qualikit --port 7861
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--port` | 服务端口号 | 7860 / 7861 |
| `--share` | 创建 Gradio 公共链接 | `False` |

### 语言切换

默认界面语言为**英文**。页面顶部有 **Language** 切换按钮，点击 **中文** 即可切换。所有标签、按钮和说明文字实时更新。

---

## QuantiKit：文本分类

QuantiKit 将文本分类全流程拆分为 6 个步骤，逐步引导操作。

### 步骤 1 · 数据上传

- **支持格式**：CSV、Excel（.xlsx/.xls）、JSON、JSONL
- 上传文件后，映射 `text`（文本列）和 `label`（标签列）
- 自动数据验证：检测缺失值、空字符串、编码问题
- **一键修复**：自动修复常见数据质量问题
- 诊断报告：标签分布、文本长度统计、重复检测

### 步骤 2 · 方法推荐

- **方法推荐引擎**：分析数据特征（数据量、类别数、不平衡比例、文本长度），推荐最优分类方法 — zero-shot、few-shot 或 fine-tuning — 并附文献引用
- **标注预算推荐**：回答"需要标注多少条？"，基于 power-law 学习曲线拟合，附 80% 置信区间和边际收益曲线
  - *冷启动模式*：基于 CSS 基准数据集（HatEval / SemEval / MFTC）的先验学习曲线
  - *经验模式*：在已标注子集上拟合 `f1 = a * n^b + c`

### 步骤 3 · 数据标注

- 内置标注界面，无需外部工具
- 逐条标注，支持**跳过**、**撤销**、**标记待审**
- 导出标注数据为 CSV，可与原始数据合并
- 进度追踪显示完成百分比

### 步骤 4 · 文本分类

三种分类方式在并行标签页中呈现：

| 子标签 | 方法 | 适用场景 |
|--------|------|---------|
| **提示词分类** | 通过 LLM API 进行 Zero/Few-shot | 小数据集（< 200 条标注） |
| **本地微调** | 本地 Transformer 微调 | 中等数据集（200+），无 API 费用 |
| **API 微调** | OpenAI 微调 API | 大数据集，最佳性能 |

**提示词分类**功能：
- Prompt 设计器：任务描述 + 类别定义 + 正面/反面示例 → 自动生成结构化提示词
- Prompt 优化器：基于 APE（自动提示工程）生成 3 个变体，在测试集上评估
- 一键全量分类

### 步骤 5 · 模型评估

- 指标：Accuracy、Macro-F1、Cohen's Kappa、各类别 Precision / Recall / F1
- 混淆矩阵可视化
- 详细分类报告

### 步骤 6 · 导出

- 下载分类结果 CSV（原始文本 + 预测标签 + 置信度）

---

## QualiKit：质性编码

QualiKit 支持访谈记录、焦点小组讨论和开放式问卷的完整质性编码流程。

### 步骤 1 · 上传与分段

- **支持格式**：纯文本（.txt）
- 自动检测说话者并分段（按段落或按说话人轮次）
- 可配置上下文窗口（包含周围句子数量）
- 分段结果表格预览

### 步骤 2 · 脱敏处理

- 自动 PII 检测：人名、邮箱、电话号码、身份证号
- 中文感知 NER：检测带称谓的中文姓名（如"张女士"、"老王"）
- 英文 NER 通过 spaCy 和 Presidio 实现
- 替换策略：假名、遮盖（`[已脱敏]`）或标签式（`[人名_1]`）
- **逐条审核**：对每个检测到的 PII 逐一接受、拒绝或编辑
- **批量操作**：全部接受、仅接受高置信度（≥ 0.90）、或将所有已接受的应用到文本

### 步骤 3 · 研究框架

- 通过可编辑交互表格定义研究问题（RQ）和子主题
- 动态添加/删除行
- **LLM 子主题建议**：连接 LLM 后端，自动分析文本并为每个 RQ 建议相关子主题
- 确认框架后进入编码

### 步骤 4 · LLM 编码

- 批量编码：LLM 逐段阅读并分配 RQ + 子主题标签，附置信度分数
- 支持 OpenAI、Anthropic、Ollama 后端
- 结果展示段落文本、分配的编码和置信度等级

### 步骤 5 · 人工审核

- 按置信度排序的审核表格
- 逐条操作：接受、拒绝或编辑（重新分配 RQ/子主题）
- 按置信度阈值批量接受
- **手动编码**：选择段落、预览内容、手动分配 RQ + 子主题
- 级联下拉菜单：子主题选项根据所选 RQ 自动过滤

### 步骤 6 · 导出

- 导出审核后的编码结果为结构化 Excel 文件

---

## 支持的 LLM 后端

| 后端 | 模型示例 | 用途 |
|------|---------|------|
| **OpenAI** | `gpt-4o`、`gpt-4o-mini`、`gpt-4.1`、`gpt-4.1-mini`、`gpt-4.1-nano` | 分类 / 编码 / Prompt 优化 |
| **Anthropic** | `claude-sonnet-4-20250514`、`claude-haiku-4-5-20251001` | 分类 / 编码 / Prompt 优化 |
| **Ollama** | `llama3`、`mistral`、`qwen2.5` | 本地推理，无需 API Key |

使用 Ollama 需先安装 [ollama.com](https://ollama.com)，然后拉取模型：

```bash
ollama pull llama3
```

---

## 示例数据集

`examples/` 目录包含即用型示例数据：

| 文件 | 模块 | 说明 |
|------|------|------|
| `sentiment_example.csv` | QuantiKit | 50 条中文商品/服务评价，3 类情感标签（正面/负面/中性） |
| `policy_example.csv` | QuantiKit | 40 条中文政策文本，8 类政策工具标签 |
| `interview_example.txt` | QualiKit | 单人社区医疗访谈记录（中文） |
| `interview_focus_group.txt` | QualiKit | 4 人焦点小组讨论 — 老年人数字服务体验（中文） |

### 实操教程：情感分类（QuantiKit）

1. 启动：`socialscikit launch` → 点击 **QuantiKit** 标签页
2. 上传 `examples/sentiment_example.csv`
3. 映射列：text → `text`，label → `label`
4. 进入**步骤 2** → 点击**推荐**查看方法建议
5. 进入**步骤 4** → 选择 LLM 后端 → 输入标签：`正面, 负面, 中性`
6. 点击**生成 Prompt** → **运行分类**
7. 进入**步骤 5** → 对比金标签评估
8. 进入**步骤 6** → 导出结果

### 实操教程：焦点小组编码（QualiKit）

1. 启动：`socialscikit launch` → 点击 **QualiKit** 标签页
2. 上传 `examples/interview_focus_group.txt`
3. **步骤 1**：选择"按说话人分段" → 点击**分段**
4. **步骤 2**：运行脱敏 → 逐条审核接受/拒绝 PII 替换
5. **步骤 3**：定义 RQ（如"数字素养障碍"）和子主题 → 可选使用 LLM 建议子主题
6. **步骤 4**：选择 LLM 后端 → 运行批量编码
7. **步骤 5**：审核结果，批量接受高置信度编码，手动修正低置信度编码
8. **步骤 6**：导出 Excel

---

## 项目结构

```
socialscikit/
├── core/                         # 共享基础设施
│   ├── data_loader.py            # 多格式数据读取（CSV/Excel/JSON/txt）
│   ├── data_validator.py         # 格式验证 + 自动修复
│   ├── data_diagnostics.py       # 数据质量诊断报告
│   ├── llm_client.py             # 统一 LLM 客户端（OpenAI/Anthropic/Ollama）
│   └── templates/                # 模板文件
│
├── quantikit/                    # 文本分类模块
│   ├── feature_extractor.py      # 数据特征提取
│   ├── method_recommender.py     # 规则推荐引擎（附文献引用）
│   ├── budget_recommender.py     # 标注预算估算
│   ├── prompt_optimizer.py       # APE 提示词生成与优化
│   ├── prompt_classifier.py      # Zero/Few-shot LLM 分类
│   ├── annotator.py              # 内置标注界面
│   ├── classifier.py             # Transformer 微调流水线
│   ├── api_finetuner.py          # OpenAI 微调 API 封装
│   └── evaluator.py              # Accuracy / F1 / Kappa / 混淆矩阵
│
├── qualikit/                     # 质性编码模块
│   ├── segmenter.py              # 文本分段（按段落/按说话人）
│   ├── segment_extractor.py      # 段落级信息提取
│   ├── deidentifier.py           # PII 检测（中文 + 英文）
│   ├── deident_reviewer.py       # 脱敏交互审核
│   ├── theme_definer.py          # 主题定义 + LLM 建议
│   ├── theme_reviewer.py         # 主题审核与重叠度检测
│   ├── coder.py                  # LLM 批量编码
│   ├── confidence_ranker.py      # 置信度评分与排序
│   ├── coding_reviewer.py        # 人机协同编码审核
│   ├── extraction_reviewer.py    # 提取结果审核
│   └── exporter.py               # Excel / Markdown 导出
│
├── ui/                           # Gradio Web 界面
│   ├── main_app.py               # 统一应用（Home + QuantiKit + QualiKit）
│   ├── quantikit_app.py          # QuantiKit UI 回调函数
│   ├── qualikit_app.py           # QualiKit UI 回调函数
│   └── i18n.py                   # 国际化（EN / ZH）
│
├── cli.py                        # 命令行入口
│
examples/                         # 示例数据集
tests/                            # 测试套件（562 个测试）
pyproject.toml                    # 包元数据与依赖
CITATION.cff                      # 引用元数据
```

---

## 关键文献

方法推荐引擎和工作流设计基于以下计算社会科学文献：

- Sun, B., Chang, C., Ang, Y. Y., Mu, R., Xu, Y. & Zhang, Z. (2026). Creation of the Chinese Adaptive Policy Communication Corpus. *ACL 2026*.
- Carlson, K. et al. (2026). The use of LLMs to annotate data in management research. *Strategic Management Journal*.
- Chae, Y. & Davidson, T. (2025). Large Language Models for text classification. *Sociological Methods & Research*.
- Do, S., Ollion, E. & Shen, R. (2024). The augmented social scientist. *Sociological Methods & Research*, 53(3).
- Dunivin, Z. O. (2024). Scalable qualitative coding with LLMs. *arXiv:2401.15170*.
- Montgomery, J. M. et al. (2024). Improving probabilistic models in text classification via active learning. *American Political Science Review*.
- Than, N. et al. (2025). Updating 'The Future of Coding'. *Sociological Methods & Research*.
- Ziems, C. et al. (2024). Can LLMs transform computational social science? *Computational Linguistics*, 50(1).
- Zhou, Y. et al. (2022). Large Language Models are human-level prompt engineers. *ICLR 2023*.

---

## 引用

如果您在研究中使用了 SocialSciKit，请引用以下论文：

```bibtex
@inproceedings{sun2026creation,
  title     = {Creation of the {Chinese} Adaptive Policy Communication Corpus},
  author    = {Sun, Bolun and Chang, Charles and Ang, Yuen Yuen and Mu, Ruotong and Xu, Yuchen and Zhang, Zhengxin},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
  year      = {2026}
}
```

---

## 开发指南

```bash
# 克隆仓库
git clone https://github.com/Baron-Sun/socialscikit.git
cd socialscikit

# 以可编辑模式安装（含开发依赖）
pip install -e ".[dev]"

# 运行全部测试
pytest tests/ -v

# 代码风格检查
ruff check .
```

### 开发模式启动

```bash
python -c "from socialscikit.ui.main_app import create_app; create_app().launch()"
```

---

## 许可证与免责声明

**许可证**：MIT

**免责声明**：

- **脱敏模块**：自动 PII 检测为初步处理工具。提交 IRB 审核前必须进行人工复核。本工具不保证完全去除所有身份信息。
- **LLM 分类/编码**：结果应视为辅助参考。重要研究结论需人工验证。
- **标注预算推荐**：基于统计估算，实际需求可能因任务复杂度和数据特征而异。

---

## 作者

**孙博伦** (Bolun Sun)

博士研究生，[凯洛格管理学院](https://www.kellogg.northwestern.edu/)，西北大学

研究方向：计算社会科学、自然语言处理、人本人工智能

邮箱：bolun.sun@kellogg.northwestern.edu | 主页：[baron-sun.github.io](https://baron-sun.github.io/)

---

## 参与贡献

本项目持续维护和更新中，欢迎提出建议、反馈问题或贡献代码！请通过 [GitHub Issues](https://github.com/Baron-Sun/socialscikit/issues) 提交，或直接发起 Pull Request。
