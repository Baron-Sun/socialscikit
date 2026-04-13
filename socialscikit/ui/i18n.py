"""Internationalization for SocialSciKit UI."""

LANGUAGES = {"en": "English", "zh": "中文"}


def t(key: str, lang: str = "en") -> str:
    """Get translated string."""
    return _T.get(key, {}).get(lang, key)


# ---------------------------------------------------------------------------
# Translation dictionary
# ---------------------------------------------------------------------------

_T = {

    # ======================================================================
    # Landing page
    # ======================================================================

    "landing.title_html": {
        "en": "# SocialSciKit",
        "zh": "# SocialSciKit",
    },
    "landing.subtitle": {
        "en": "Zero-code text analysis & research methods toolkit for social science researchers",
        "zh": "面向社会科学研究者的零代码文本分析与研究方法工具包",
    },
    "landing.quantikit_card": {
        "en": (
            "### QuantiKit — Text Classification\n\n"
            "**Use cases:** Sentiment analysis, stance detection, frame analysis, topic classification\n\n"
            "**Workflow:** Upload data \u2192 Method recommendation \u2192 Annotation \u2192 Classification \u2192 Evaluation \u2192 Export\n\n"
            "**Features:**\n"
            "- Method recommendation engine (with CSS literature references)\n"
            "- Annotation budget recommendation (learning curve fitting)\n"
            "- Zero-shot / Few-shot / Fine-tuning\n"
            "- Prompt multi-version auto-optimization\n"
        ),
        "zh": (
            "### QuantiKit \u2014 文本分类\n\n"
            "**适用场景：** 情感分析、立场检测、框架分析、主题分类\n\n"
            "**流程：** 上传数据 \u2192 方法推荐 \u2192 标注 \u2192 分类 \u2192 评估 \u2192 导出\n\n"
            "**能力：**\n"
            "- 方法推荐引擎（附 CSS 文献依据）\n"
            "- 标注预算推荐（学习曲线拟合）\n"
            "- Zero-shot / Few-shot / Fine-tuning\n"
            "- Prompt 多版本自动优化\n"
        ),
    },
    "landing.qualikit_card": {
        "en": (
            "### QualiKit — Qualitative Coding\n\n"
            "**Use cases:** Interview analysis, open-ended surveys, thematic coding\n\n"
            "**Workflow:** Upload data \u2192 De-identification \u2192 Theme definition \u2192 Coding \u2192 Export\n\n"
            "**Features:**\n"
            "- PII auto-detection with interactive review\n"
            "- AI-assisted theme suggestion (TF-IDF / LLM)\n"
            "- LLM multi-label coding + confidence ranking\n"
            "- Excerpt table + co-occurrence matrix + analysis memo\n"
        ),
        "zh": (
            "### QualiKit \u2014 质性编码\n\n"
            "**适用场景：** 访谈分析、开放式问卷、文献主题编码\n\n"
            "**流程：** 上传数据 \u2192 脱敏 \u2192 主题定义 \u2192 编码 \u2192 导出\n\n"
            "**能力：**\n"
            "- PII 自动检测与交互审核\n"
            "- AI 辅助主题建议（TF-IDF / LLM）\n"
            "- LLM 多标签编码 + 置信度分级\n"
            "- 摘录表 + 共现矩阵 + 分析备忘录\n"
        ),
    },
    "landing.toolbox_card": {
        "en": (
            "### Toolbox — Research Methods Tools\n\n"
            "**Standalone tools** that work independently or together with QuantiKit / QualiKit:\n\n"
            "- **ICR Calculator** — Inter-coder reliability (Cohen's Kappa, Krippendorff's Alpha, multi-label Jaccard); supports 2+ coders with auto metric selection\n"
            "- **Consensus Coding** — Multi-LLM majority-vote coding with 2\u20135 configurable LLM backends\n"
            "- **Methods Generator** — Auto-generate methods section paragraphs (EN/ZH) from pipeline logs or manual input\n"
        ),
        "zh": (
            "### 工具箱 — 研究方法工具\n\n"
            "**独立工具**，可单独使用或与 QuantiKit / QualiKit 搭配：\n\n"
            "- **ICR 计算器** — 编码者间信度（Cohen's Kappa、Krippendorff's Alpha、多标签 Jaccard）；支持 2+ 编码者自动选择指标\n"
            "- **共识编码** — 多 LLM 多数投票编码，支持 2\u20135 个可配置的 LLM 后端\n"
            "- **方法论生成器** — 从流水线日志或手动输入自动生成方法部分段落（中英双语）\n"
        ),
    },
    "landing.quickstart": {
        "en": (
            "### Quick Start\n\n"
            "1. Click the **QuantiKit**, **QualiKit**, or **Toolbox** tab above to enter the corresponding module\n"
            "2. Follow the numbered steps in order \u2014 results at each step can be reviewed and edited before proceeding\n"
            "3. When LLM features are needed, provide an API Key (OpenAI / Anthropic) or use Ollama for local inference\n"
            "4. Use the **Toolbox** for standalone tools: ICR calculation, multi-LLM consensus coding, or auto-generating methods sections"
        ),
        "zh": (
            "### 快速开始\n\n"
            "1. 点击上方 **QuantiKit**、**QualiKit** 或 **工具箱** 标签页进入对应模块\n"
            "2. 按步骤编号依次操作 \u2014 每步结果可审核编辑，确认后再进入下一步\n"
            "3. 需要 LLM 功能时提供 API Key（OpenAI / Anthropic），或使用 Ollama 本地推理\n"
            "4. 使用**工具箱**可独立进行：ICR 信度计算、多 LLM 共识编码、自动生成方法论段落"
        ),
    },
    "landing.examples": {
        "en": (
            "**Example data:**\n"
            "`examples/sentiment_example.csv` (sentiment classification \u00b7 QuantiKit) \u00b7 "
            "`examples/policy_example.csv` (policy instrument classification \u00b7 QuantiKit) \u00b7 "
            "`examples/interview_example.txt` (single interview \u00b7 QualiKit) \u00b7 "
            "`examples/interview_focus_group.txt` (focus group \u00b7 QualiKit) \u00b7 "
            "`examples/icr_example.csv` (inter-coder reliability \u00b7 Toolbox) \u00b7 "
            "`examples/consensus_example.csv` (consensus coding \u00b7 Toolbox) \u00b7 "
            "`examples/methods_log_quantikit.json` / `methods_log_qualikit.json` (methods generator \u00b7 Toolbox)"
        ),
        "zh": (
            "**示例数据：**\n"
            "`examples/sentiment_example.csv`（情感分类 \u00b7 QuantiKit）\u00b7 "
            "`examples/policy_example.csv`（政策工具分类 \u00b7 QuantiKit）\u00b7 "
            "`examples/interview_example.txt`（单人访谈 \u00b7 QualiKit）\u00b7 "
            "`examples/interview_focus_group.txt`（焦点小组 \u00b7 QualiKit）\u00b7 "
            "`examples/icr_example.csv`（编码者间信度 \u00b7 工具箱）\u00b7 "
            "`examples/consensus_example.csv`（共识编码 \u00b7 工具箱）\u00b7 "
            "`examples/methods_log_quantikit.json` / `methods_log_qualikit.json`（方法论生成 \u00b7 工具箱）"
        ),
    },
    "landing.references": {
        "en": (
            "<details>\n"
            '<summary style="cursor:pointer; font-weight:600; color:#2c3e50;">References</summary>\n\n'
            "- Ziems, C. et al. (2024). Can LLMs transform computational social science? *Computational Linguistics*, 50(1).\n"
            "- Chae, Y. & Davidson, T. (2025). LLMs for text classification. *Sociological Methods & Research*.\n"
            "- Dunivin, Z. O. (2024). Scalable qualitative coding with LLMs. *arXiv:2401.15170*.\n"
            "- Do, S. et al. (2024). The augmented social scientist. *SMR*, 53(3).\n"
            "- Zhou, Y. et al. (2022). Large Language Models Are Human-Level Prompt Engineers. *ICLR 2023*.\n\n"
            "</details>"
        ),
        "zh": (
            "<details>\n"
            '<summary style="cursor:pointer; font-weight:600; color:#2c3e50;">参考文献</summary>\n\n'
            "- Ziems, C. et al. (2024). Can LLMs transform computational social science? *Computational Linguistics*, 50(1).\n"
            "- Chae, Y. & Davidson, T. (2025). LLMs for text classification. *Sociological Methods & Research*.\n"
            "- Dunivin, Z. O. (2024). Scalable qualitative coding with LLMs. *arXiv:2401.15170*.\n"
            "- Do, S. et al. (2024). The augmented social scientist. *SMR*, 53(3).\n"
            "- Zhou, Y. et al. (2022). Large Language Models Are Human-Level Prompt Engineers. *ICLR 2023*.\n\n"
            "</details>"
        ),
    },

    # ======================================================================
    # QualiKit Step 1 - Upload & Segment
    # ======================================================================

    "ql.s1.tab": {
        "en": "Step 1 \u00b7 Upload & Segment",
        "zh": "Step 1 \u00b7 上传与分段",
    },
    "ql.s1.title": {
        "en": (
            "## Upload & Segment\n"
            "Upload a plain-text interview transcript, choose a segmentation mode, and preview the segments."
        ),
        "zh": (
            "## 上传与分段\n"
            "上传纯文本访谈记录，选择分段模式，预览分段结果。"
        ),
    },
    "ql.s1.upload": {
        "en": "Upload text file",
        "zh": "上传文本文件",
    },
    "ql.s1.download_example": {
        "en": "Download example",
        "zh": "下载示例",
    },
    "ql.s1.example_file": {
        "en": "Example file",
        "zh": "示例文件",
    },
    "ql.s1.preview": {
        "en": "Text preview",
        "zh": "原文预览",
    },
    "ql.s1.preview_placeholder": {
        "en": "Text will appear here after upload...",
        "zh": "上传文件后将在此显示原文……",
    },
    "ql.s1.segment_title": {
        "en": "### Segmentation Settings",
        "zh": "### 分段设置",
    },
    "ql.s1.mode": {
        "en": "Segmentation mode",
        "zh": "分段模式",
    },
    "ql.s1.mode_info": {
        "en": "paragraph=by paragraph | sentence=by sentence | context_window=core sentence \u00b1 context",
        "zh": "paragraph=按段落 | sentence=按句子 | context_window=核心句\u00b1上下文",
    },
    "ql.s1.context_window": {
        "en": "Context window \u00b1N sentences",
        "zh": "上下文窗口 \u00b1N 句",
    },
    "ql.s1.context_info": {
        "en": "Only effective in context_window mode",
        "zh": "仅 context_window 模式有效",
    },
    "ql.s1.segment_btn": {
        "en": "Preview segments",
        "zh": "预览分段",
    },
    "ql.s1.segment_preview": {
        "en": "Segment preview",
        "zh": "分段预览",
    },

    # ======================================================================
    # QualiKit Step 2 - De-identification
    # ======================================================================

    "ql.s2.tab": {
        "en": "Step 2 \u00b7 De-identification",
        "zh": "Step 2 \u00b7 脱敏",
    },
    "ql.s2.title": {
        "en": (
            "## PII De-identification\n"
            "Detect and replace personally identifiable information (PII) in segmented text before thematic coding.\n\n"
            "> **Methodological basis:**\n"
            "> - De-identification is a fundamental requirement of qualitative research ethics "
            "and must be completed before analysis (Kaiser, 2009; Saunders et al., 2015)\n"
            "> - This tool uses rule-based NER detection with configurable replacement strategies "
            "(Dernoncourt et al., 2017)\n"
            "> - Automated de-identification is only a preliminary tool; manual review is essential "
            "before IRB submission (Surmiak, 2018)\n\n"
            "**Replacement strategy descriptions:**\n"
            "- `placeholder`: Replace with `[PERSON_1]`, `[ORG_2]`, etc. "
            "(recommended \u2014 preserves readability and is reversible)\n"
            "- `category`: Replace with entity category names, e.g. `[Person Name]`\n"
            "- `redact`: Replace with `[REDACTED]` for complete masking"
        ),
        "zh": (
            "## PII 脱敏\n"
            "在主题编码前对分段文本进行个人身份信息 (PII) 检测与替换。\n\n"
            "> **方法论依据：**\n"
            "> - 脱敏是质性研究伦理的基本要求，需在分析前完成"
            " (Kaiser, 2009; Saunders et al., 2015)\n"
            "> - 本工具采用基于规则的 NER 检测 + 可配置替换策略"
            " (Dernoncourt et al., 2017)\n"
            "> - 自动脱敏仅为初步工具，IRB 审核前务必人工复核"
            " (Surmiak, 2018)\n\n"
            "**替换策略说明：**\n"
            "- `placeholder`：用 `[PERSON_1]`、`[ORG_2]` 等占位符替换"
            "（推荐，保留可读性且可逆）\n"
            "- `category`：用实体类别名替换，如 `[人名]`\n"
            "- `redact`：用 `[REDACTED]` 完全遮盖"
        ),
    },
    "ql.s2.entity_types": {
        "en": "Entity types to detect",
        "zh": "检测实体类型",
    },
    "ql.s2.strategy": {
        "en": "Replacement strategy",
        "zh": "替换策略",
    },
    "ql.s2.run_btn": {
        "en": "Run de-identification",
        "zh": "运行脱敏",
    },
    "ql.s2.stats": {
        "en": "Detection statistics",
        "zh": "检测统计",
    },
    "ql.s2.progress": {
        "en": "Review progress",
        "zh": "审阅进度",
    },
    "ql.s2.review_table": {
        "en": "Detection results \u2014 review each item",
        "zh": "检测结果 \u2014 逐条审阅",
    },
    "ql.s2.detail_title": {
        "en": "**Selected item detail** \u2014 Enter index to view context, accept or reject each item",
        "zh": "**选中项详情** \u2014 输入序号查看上下文，逐条决定接受或拒绝",
    },
    "ql.s2.index": {
        "en": "Index",
        "zh": "序号",
    },
    "ql.s2.accept": {
        "en": "\u2705 Accept",
        "zh": "\u2705 接受",
    },
    "ql.s2.reject": {
        "en": "\u274c Reject",
        "zh": "\u274c 拒绝",
    },
    "ql.s2.detail": {
        "en": "Replacement detail",
        "zh": "替换详情",
    },
    "ql.s2.custom_text": {
        "en": "Custom replacement text",
        "zh": "自定义替换文本",
    },
    "ql.s2.custom_placeholder": {
        "en": "[Respondent A]",
        "zh": "[受访者A]",
    },
    "ql.s2.edit": {
        "en": "\u270f\ufe0f Edit",
        "zh": "\u270f\ufe0f 编辑",
    },
    "ql.s2.bulk_title": {
        "en": "### Batch Operations",
        "zh": "### 批量操作",
    },
    "ql.s2.accept_all": {
        "en": "Accept all",
        "zh": "全部接受",
    },
    "ql.s2.accept_high": {
        "en": "Accept high confidence only (>0.9)",
        "zh": "仅接受高置信度 (>0.9)",
    },
    "ql.s2.apply": {
        "en": "Apply to segments",
        "zh": "应用到段落",
    },

    # ======================================================================
    # QualiKit Step 3 - Research Framework
    # ======================================================================

    "ql.s3.tab": {
        "en": "Step 3 \u00b7 Research Framework",
        "zh": "Step 3 \u00b7 研究框架",
    },
    "ql.s3.title": {
        "en": (
            "## Define Research Framework\n"
            "Define research questions (RQ) and sub-themes in the tables below.\n"
            "A segment can match multiple RQs and multiple sub-themes simultaneously.\n\n"
            "> **Tip:** Sub-themes can be left blank and the LLM will generate labels automatically. "
            "If sub-themes are pre-defined, the LLM will only choose from them and will not "
            "generate new labels."
        ),
        "zh": (
            "## 定义研究框架\n"
            "在下方表格中定义研究问题 (RQ) 和子主题。\n"
            "一个段落可以同时匹配多个 RQ 和多个子主题。\n\n"
            "> **提示：** 子主题可以留空，LLM 将自动生成标签。"
            "若预先定义子主题，LLM 编码时只会从中选择，不会"
            "生成新标签。"
        ),
    },
    "ql.s3.rq_title": {
        "en": "### Research Questions",
        "zh": "### 研究问题",
    },
    "ql.s3.rq_table": {
        "en": "Research Questions (click + to add rows)",
        "zh": "研究问题（点击 + 添加更多行）",
    },
    "ql.s3.rq_headers": {
        "en": ["RQ ID", "Description"],
        "zh": ["RQ编号", "描述"],
    },
    "ql.s3.sub_title": {
        "en": "### Sub-themes (Optional)",
        "zh": "### 子主题（可选）",
    },
    "ql.s3.sub_table": {
        "en": "Sub-themes (parent RQ must match IDs above)",
        "zh": "子主题（所属RQ 必须与上表编号一致）",
    },
    "ql.s3.sub_headers": {
        "en": ["Parent RQ", "Sub-theme Name"],
        "zh": ["所属RQ", "子主题名称"],
    },
    "ql.s3.confirm_btn": {
        "en": "Confirm Framework \u2714",
        "zh": "确认研究框架 \u2714",
    },
    "ql.s3.parsed": {
        "en": "Parse result",
        "zh": "解析结果",
    },
    "ql.s3.suggest_title": {
        "en": (
            "### AI Sub-theme Suggestions (Optional)\n"
            "Based on segmented text, let the LLM automatically suggest sub-theme categories "
            "for reference when filling in the table above."
        ),
        "zh": (
            "### AI 子主题建议（可选）\n"
            "基于已分段文本，让 LLM 自动建议子主题分类，供参考填入上方表格。"
        ),
    },
    "ql.s3.backend": {
        "en": "LLM Backend",
        "zh": "LLM 后端",
    },
    "ql.s3.model": {
        "en": "Model",
        "zh": "模型",
    },
    "ql.s3.api_key": {
        "en": "API Key",
        "zh": "API Key",
    },
    "ql.s3.suggest_btn": {
        "en": "Generate sub-theme suggestions",
        "zh": "生成子主题建议",
    },
    "ql.s3.suggest_result": {
        "en": "Suggestion results",
        "zh": "建议结果",
    },

    # ======================================================================
    # QualiKit Step 4 - LLM Coding
    # ======================================================================

    "ql.s4.tab": {
        "en": "Step 4 \u00b7 LLM Coding",
        "zh": "Step 4 \u00b7 LLM 编码",
    },
    "ql.s4.title": {
        "en": (
            "## LLM Coding\n"
            "Use an LLM to match each segment against the research framework defined in Step 3.\n"
            "A segment can belong to multiple RQs and sub-themes simultaneously (multi-label classification)."
        ),
        "zh": (
            "## LLM 编码\n"
            "使用 LLM 将每个段落与 Step 3 中定义的研究框架匹配。\n"
            "一个段落可以同时属于多个 RQ 和子主题（多标签分类）。"
        ),
    },
    "ql.s4.backend": {
        "en": "LLM Backend",
        "zh": "LLM 后端",
    },
    "ql.s4.model": {
        "en": "Model",
        "zh": "模型",
    },
    "ql.s4.api_key": {
        "en": "API Key",
        "zh": "API Key",
    },
    "ql.s4.run_btn": {
        "en": "Start coding",
        "zh": "开始编码",
    },
    "ql.s4.result": {
        "en": "Coding results",
        "zh": "编码结果",
    },
    "ql.s4.detail": {
        "en": "Coding details",
        "zh": "编码详情",
    },

    # ======================================================================
    # QualiKit Step 5 - Review
    # ======================================================================

    "ql.s5.tab": {
        "en": "Step 5 \u00b7 Review",
        "zh": "Step 5 \u00b7 人工审阅",
    },
    "ql.s5.title": {
        "en": (
            "## Review LLM Coding Results\n"
            "Select an index in the table; the full text and source location will appear below.\n"
            "Accept / reject / edit each item individually, or batch-accept high-confidence results."
        ),
        "zh": (
            "## 审阅 LLM 编码结果\n"
            "在表格中选择序号，下方自动显示完整文本和原文定位。\n"
            "逐条接受 / 拒绝 / 编辑，或批量接受高置信度结果。"
        ),
    },
    "ql.s5.stats": {
        "en": "Review progress",
        "zh": "审阅进度",
    },
    "ql.s5.table": {
        "en": "Coding results list",
        "zh": "编码结果列表",
    },
    "ql.s5.detail_title": {
        "en": "**Selected item detail** \u2014 Enter index to view full text and source location",
        "zh": "**选中项详情** \u2014 输入序号后自动显示完整文本和原文定位",
    },
    "ql.s5.index": {
        "en": "Index",
        "zh": "序号",
    },
    "ql.s5.accept": {
        "en": "\u2705 Accept",
        "zh": "\u2705 接受",
    },
    "ql.s5.reject": {
        "en": "\u274c Reject",
        "zh": "\u274c 拒绝",
    },
    "ql.s5.detail": {
        "en": "Segment detail and source location",
        "zh": "段落详情与原文定位",
    },
    "ql.s5.edit_rq": {
        "en": "Modify RQ",
        "zh": "修改 RQ",
    },
    "ql.s5.edit_sub": {
        "en": "Modify sub-theme",
        "zh": "修改子主题",
    },
    "ql.s5.edit": {
        "en": "\u270f\ufe0f Edit",
        "zh": "\u270f\ufe0f 编辑",
    },
    "ql.s5.bulk_title": {
        "en": "### Batch Operations",
        "zh": "### 批量操作",
    },
    "ql.s5.threshold": {
        "en": "Confidence threshold",
        "zh": "置信度阈值",
    },
    "ql.s5.bulk_accept": {
        "en": "Batch accept high confidence",
        "zh": "批量接受高置信度",
    },
    "ql.s5.manual_title": {
        "en": "### Manually add missed segments",
        "zh": "### 手动添加遗漏段落",
    },
    "ql.s5.manual_preview_default": {
        "en": '<p style="color:#888;padding:0.5rem;">Enter segment ID to preview content.</p>',
        "zh": '<p style="color:#888;padding:0.5rem;">输入段落 ID 后自动预览内容。</p>',
    },
    "ql.s5.seg_id": {
        "en": "Segment ID",
        "zh": "段落 ID",
    },
    "ql.s5.rq_label": {
        "en": "RQ Label",
        "zh": "RQ 标签",
    },
    "ql.s5.sub_theme": {
        "en": "Sub-theme",
        "zh": "子主题",
    },
    "ql.s5.add": {
        "en": "Add",
        "zh": "添加",
    },

    # ======================================================================
    # QualiKit Step 6 - Export
    # ======================================================================

    "ql.s6.tab": {
        "en": "Step 6 \u00b7 Export",
        "zh": "Step 6 \u00b7 导出",
    },
    "ql.s6.title": {
        "en": (
            "## Export Review Results\n"
            "- **Excel:** Contains text, RQ, sub-theme, confidence, position, review status\n"
            "- Rejected items are excluded from export"
        ),
        "zh": (
            "## 导出审阅结果\n"
            "- **Excel：** 含文本、RQ、子主题、置信度、位置、审核状态\n"
            "- 被拒绝的条目不会导出"
        ),
    },
    "ql.s6.export_btn": {
        "en": "Export Excel",
        "zh": "导出 Excel",
    },
    "ql.s6.file": {
        "en": "Excel file",
        "zh": "Excel 文件",
    },

    # ======================================================================
    # Callback messages
    # ======================================================================

    "msg.upload_first": {
        "en": "Please upload a text file first.",
        "zh": "请上传文本文件。",
    },
    "msg.segment_first": {
        "en": "Please complete segmentation in Step 1 first.",
        "zh": "请先在 Step 1 中完成分段。",
    },
    "msg.define_rq_first": {
        "en": "Please define the research framework in Step 3 first.",
        "zh": "请先在 Step 3 中定义研究问题。",
    },
    "msg.run_deident_first": {
        "en": "Please run de-identification first.",
        "zh": "请先运行脱敏。",
    },
    "msg.run_extraction_first": {
        "en": "Please run extraction first.",
        "zh": "请先运行提取。",
    },
    "msg.enter_api_key": {
        "en": "Please enter an API Key.",
        "zh": "请输入 API Key。",
    },
    "msg.invalid_index": {
        "en": "Please enter a valid index.",
        "zh": "请输入有效的序号。",
    },
    "msg.no_data": {
        "en": "No data to export.",
        "zh": "无可导出数据。",
    },
    "msg.at_least_one_rq": {
        "en": "Please define at least one research question.",
        "zh": "请至少定义一个研究问题。",
    },
    "msg.operation_failed": {
        "en": "Operation failed: {}",
        "zh": "操作失败：{}",
    },
    "msg.accepted_n": {
        "en": "Accepted {} items.",
        "zh": "已接受 {} 项。",
    },
    "msg.accepted_high_n": {
        "en": "Accepted {} high confidence items.",
        "zh": "已接受 {} 项高置信度替换。",
    },
    "msg.applied_deident": {
        "en": "De-identification applied, {} segment texts updated.",
        "zh": "已应用脱敏，{} 个段落文本已更新。",
    },
    "msg.loaded_chars": {
        "en": "Loaded: {} characters, {} lines",
        "zh": "已加载：{} 字符，{} 行",
    },
    "msg.segment_done": {
        "en": "Segmentation complete: {} segments ({})",
        "zh": "分段完成：共 {} 个段落（{}）",
    },
    "msg.no_pii": {
        "en": "No PII entities detected.",
        "zh": "未检出任何 PII 实体。",
    },
    "msg.detected_entities": {
        "en": "Detected entities: {}",
        "zh": "检出实体：{}",
    },
    "msg.export_success": {
        "en": "Exported {} records to Excel.",
        "zh": "已导出 {} 条记录至 Excel。",
    },
    "msg.framework_confirmed": {
        "en": "Confirmed {} research questions:",
        "zh": "已确认 {} 个研究问题：",
    },
    "msg.sub_themes_auto": {
        "en": "(auto-generated)",
        "zh": "（待生成）",
    },
    "msg.enter_replacement": {
        "en": "Please enter replacement text.",
        "zh": "请输入替换文本。",
    },
    "msg.cannot_apply": {
        "en": "Cannot apply.",
        "zh": "无法应用。",
    },
    "msg.no_content": {
        "en": "No content to display. Please run extraction first.",
        "zh": "无可显示内容。请先运行提取。",
    },
    "msg.upload_data_first": {
        "en": "Please upload a data file first.",
        "zh": "请上传数据文件。",
    },
    "msg.upload_data": {
        "en": "Please upload data first.",
        "zh": "请先上传数据。",
    },
    "msg.create_session_first": {
        "en": "Please create an annotation session first.",
        "zh": "请先创建标注会话。",
    },
    "msg.enter_label": {
        "en": "Please enter a label.",
        "zh": "请输入标签。",
    },
    "msg.undone": {
        "en": "Undone.",
        "zh": "已撤销。",
    },
    "msg.nothing_to_undo": {
        "en": "Nothing to undo.",
        "zh": "没有可撤销的操作。",
    },
    "msg.all_annotated": {
        "en": "(All items annotated)",
        "zh": "（已完成全部标注）",
    },
    "msg.merged_n": {
        "en": "Merged {} annotations into the '{}' column of the main dataset.",
        "zh": "已将 {} 条标注合并到主数据集的 '{}' 列。",
    },
    "msg.enter_classes": {
        "en": "Please enter class labels (comma-separated).",
        "zh": "请输入类别标签（逗号分隔）。",
    },
    "msg.enter_text_col": {
        "en": "Please specify the text column name.",
        "zh": "请指定文本列名。",
    },
    "msg.all_labeled": {
        "en": "All texts already have labels; no classification needed.",
        "zh": "所有文本都已有标签，无需分类。",
    },
    "msg.need_label_col": {
        "en": "Label column is needed for evaluation. Please ensure the data contains ground-truth labels.",
        "zh": "需要标签列进行评估。请确保数据中已有真实标签。",
    },
    "msg.run_classification_first": {
        "en": "Please run classification first.",
        "zh": "请先运行分类。",
    },
    "msg.col_mapping_confirmed": {
        "en": "Column mapping confirmed",
        "zh": "列映射已确认",
    },
    "msg.select_valid_text_col": {
        "en": "Please select a valid text column",
        "zh": "请选择有效的文本列",
    },
    "msg.fixed": {
        "en": "Fixed. {} ({} rows remaining)",
        "zh": "已修复。{}（剩余 {} 行）",
    },
    "msg.lock_framework_first": {
        "en": "Please confirm the research framework first.",
        "zh": "请先锁定主题框架。",
    },
    "msg.complete_review_first": {
        "en": "Please complete the review first.",
        "zh": "请先完成审阅。",
    },
    "msg.generate_prompt_first": {
        "en": "Please generate or enter a Prompt above first.",
        "zh": "请先在上方生成或输入 Prompt。",
    },
    "msg.variant_empty": {
        "en": "{} is empty and cannot be adopted.",
        "zh": "{} 为空，无法采用。",
    },
    "msg.adopted_variant": {
        "en": "Adopted {} as the current Prompt",
        "zh": "已采用{}作为当前 Prompt",
    },
    "msg.enter_openai_key": {
        "en": "Please enter an OpenAI API Key.",
        "zh": "请输入 OpenAI API Key。",
    },

    # ======================================================================
    # QuantiKit Step 1 - Data Upload
    # ======================================================================

    "qt.s1.tab": {
        "en": "Step 1 \u00b7 Data Upload",
        "zh": "Step 1 \u00b7 数据上传",
    },
    "qt.s1.title": {
        "en": (
            "## Upload Data\n"
            "Upload a CSV or Excel file. The system will automatically validate the format and generate a diagnostic report."
        ),
        "zh": (
            "## 上传数据\n"
            "上传 CSV 或 Excel 文件，系统将自动验证格式并生成诊断报告。"
        ),
    },
    "qt.s1.file": {
        "en": "Select file",
        "zh": "选择文件",
    },
    "qt.s1.download_template": {
        "en": "Download template",
        "zh": "下载模板",
    },
    "qt.s1.template": {
        "en": "Template",
        "zh": "模板",
    },
    "qt.s1.col_mapping_title": {
        "en": "### Column Mapping\nSelect the text column and label column in your data, then click confirm to re-run diagnostics.",
        "zh": "### 列映射\n选择数据中的文本列和标签列，点击确认后系统将重新诊断。",
    },
    "qt.s1.text_col": {
        "en": "Text column",
        "zh": "文本列",
    },
    "qt.s1.label_col": {
        "en": "Label column",
        "zh": "标签列",
    },
    "qt.s1.confirm_col": {
        "en": "Confirm column mapping",
        "zh": "确认列映射",
    },
    "qt.s1.report": {
        "en": "Diagnostic report",
        "zh": "诊断报告",
    },
    "qt.s1.issues": {
        "en": "Detected issues",
        "zh": "检测到的问题",
    },
    "qt.s1.preview": {
        "en": "Data preview",
        "zh": "数据预览",
    },
    "qt.s1.fix_btn": {
        "en": "Auto-fix",
        "zh": "一键修复",
    },

    # ======================================================================
    # QuantiKit Step 2 - Method Recommendation
    # ======================================================================

    "qt.s2.tab": {
        "en": "Step 2 \u00b7 Method Recommendation",
        "zh": "Step 2 \u00b7 方法推荐",
    },
    "qt.s2.title": {
        "en": (
            "## Method Recommendation & Annotation Budget\n"
            "The system recommends the most suitable classification method based on data characteristics "
            "and estimates the required annotation volume."
        ),
        "zh": (
            "## 方法推荐与标注预算\n"
            "系统根据数据特征推荐最适合的分类方法，并估算所需标注量。"
        ),
    },
    "qt.s2.task_type": {
        "en": "Task type",
        "zh": "任务类型",
    },
    "qt.s2.num_classes": {
        "en": "Number of classes",
        "zh": "类别数",
    },
    "qt.s2.target_f1": {
        "en": "Target F1",
        "zh": "目标 F1",
    },
    "qt.s2.budget": {
        "en": "Budget",
        "zh": "预算",
    },
    "qt.s2.recommend_btn": {
        "en": "Generate recommendation",
        "zh": "生成推荐",
    },
    "qt.s2.features": {
        "en": "Task features",
        "zh": "任务特征",
    },
    "qt.s2.annotation_budget": {
        "en": "Annotation budget",
        "zh": "标注预算",
    },
    "qt.s2.recommendation": {
        "en": "Recommended approach",
        "zh": "推荐方案",
    },
    "qt.s2.curve_plot": {
        "en": "Marginal return curve",
        "zh": "边际收益曲线",
    },
    "qt.s2.key_points": {
        "en": "Key points",
        "zh": "关键节点",
    },

    # ======================================================================
    # QuantiKit Step 3 - Annotation
    # ======================================================================

    "qt.s3.tab": {
        "en": "Step 3 \u00b7 Annotation",
        "zh": "Step 3 \u00b7 标注",
    },
    "qt.s3.title": {
        "en": (
            "## Manual Annotation (Optional)\n"
            "Provide a small set of labeled data for few-shot or fine-tuning. "
            "The text column and label column are set in Step 1.\n\n"
            "**If your data already has sufficient labeled data, you can skip directly to "
            "Step 4 Classification or Fine-tuning without manual annotation.**"
        ),
        "zh": (
            "## 人工标注（可选）\n"
            "为 few-shot 或 fine-tuning 提供少量标注数据。文本列和标签列已在 Step 1 中设定。\n\n"
            "**如果你的数据已有足够的标注数据，可以直接跳到 Step 4 分类或 Fine-tuning，无需再手动标注。**"
        ),
    },
    "qt.s3.labels": {
        "en": "Label list",
        "zh": "标签列表",
    },
    "qt.s3.labels_placeholder": {
        "en": "comma-separated",
        "zh": "逗号分隔",
    },
    "qt.s3.shuffle": {
        "en": "Random order",
        "zh": "随机顺序",
    },
    "qt.s3.create_session": {
        "en": "Create annotation session",
        "zh": "创建标注会话",
    },
    "qt.s3.progress": {
        "en": "Progress",
        "zh": "进度",
    },
    "qt.s3.current_pos": {
        "en": "Current position",
        "zh": "当前位置",
    },
    "qt.s3.text_to_annotate": {
        "en": "Text to annotate",
        "zh": "待标注文本",
    },
    "qt.s3.label": {
        "en": "Label",
        "zh": "标签",
    },
    "qt.s3.annotate_btn": {
        "en": "Annotate",
        "zh": "标注",
    },
    "qt.s3.skip_btn": {
        "en": "Skip",
        "zh": "跳过",
    },
    "qt.s3.note": {
        "en": "Note",
        "zh": "备注",
    },
    "qt.s3.note_placeholder": {
        "en": "Optional: note any concerns",
        "zh": "可选：标记疑问原因",
    },
    "qt.s3.flag_btn": {
        "en": "Flag",
        "zh": "标记",
    },
    "qt.s3.undo_btn": {
        "en": "Undo",
        "zh": "撤销",
    },
    "qt.s3.export_accordion": {
        "en": "Export & Merge",
        "zh": "导出与合并",
    },
    "qt.s3.include_skipped": {
        "en": "Include skipped/flagged items",
        "zh": "包含跳过/标记项",
    },
    "qt.s3.preview_btn": {
        "en": "Preview",
        "zh": "预览",
    },
    "qt.s3.download_csv": {
        "en": "Download CSV",
        "zh": "下载 CSV",
    },
    "qt.s3.merge_btn": {
        "en": "Merge into main data",
        "zh": "合并到主数据",
    },
    "qt.s3.results": {
        "en": "Annotation results",
        "zh": "标注结果",
    },
    "qt.s3.download_file": {
        "en": "Download file",
        "zh": "下载文件",
    },

    # ======================================================================
    # QuantiKit Step 4 - Classification
    # ======================================================================

    "qt.s4.tab": {
        "en": "Step 4 \u00b7 Classification",
        "zh": "Step 4 \u00b7 分类",
    },
    "qt.s4.title": {
        "en": (
            "## Text Classification\n"
            "Classify text using LLM Prompts or local Fine-tuning. "
            "The text column and label column are set in Step 1."
        ),
        "zh": (
            "## 文本分类\n"
            "通过 LLM Prompt 或本地 Fine-tuning 对文本进行分类。文本列和标签列已在 Step 1 中设定。"
        ),
    },
    "qt.s4.backend": {
        "en": "LLM Backend",
        "zh": "LLM 后端",
    },
    "qt.s4.model": {
        "en": "Model",
        "zh": "模型",
    },
    "qt.s4.api_key": {
        "en": "API Key",
        "zh": "API Key",
    },
    "qt.s4.classes": {
        "en": "Classes",
        "zh": "类别",
    },
    "qt.s4.classes_placeholder": {
        "en": "comma-separated",
        "zh": "逗号分隔",
    },

    # -- Prompt classification sub-tab --

    "qt.s4.prompt_tab": {
        "en": "Prompt Classification",
        "zh": "Prompt 分类",
    },
    "qt.s4.design_title": {
        "en": (
            "### Step 1: Design Prompt\n"
            "Describe your classification task, fill in class definitions and examples. "
            "The system will call an LLM to generate a high-quality classification prompt.\n"
            "You can also skip this step and write directly in the \"Current Prompt\" box."
        ),
        "zh": (
            "### 第一步：设计 Prompt\n"
            "描述你的分类任务，填写类别定义和示例。系统将调用 LLM 为你生成高质量的分类 Prompt。\n"
            "也可以跳过此步，直接在「当前 Prompt」框中手写。"
        ),
    },
    "qt.s4.task_desc": {
        "en": "Task description (describe what you want to do in natural language)",
        "zh": "任务描述（用自然语言描述你要做什么）",
    },
    "qt.s4.task_desc_placeholder": {
        "en": (
            "e.g.: I need to classify paragraphs in policy documents to determine whether each "
            "paragraph discusses W (welfare), N (neutral), or R (rights) issues..."
        ),
        "zh": (
            "例如：我需要对政策文档中的段落进行分类，判断每个段落讨论的是 W（福利）、"
            "N（中性）还是 R（权利）类议题。段落来自中国政府工作报告……"
        ),
    },
    "qt.s4.class_defs": {
        "en": "Class definitions (one per line, format: label: definition)",
        "zh": "类别定义（每行一个，格式：标签: 定义 或 标签：定义）",
    },
    "qt.s4.class_defs_placeholder": {
        "en": (
            "W: Related to social welfare, security, assistance, etc.\n"
            "N: Neutral or not related to specific policy positions\n"
            "R: Related to civil rights, rule of law, etc."
        ),
        "zh": (
            "W：涉及社会福利、保障、救助等内容\n"
            "N：中性或不涉及特定政策立场\n"
            "R：涉及公民权利、法治等内容"
        ),
    },
    "qt.s4.positive_ex": {
        "en": "Positive examples (optional \u2014 one per line, format: label: example text)",
        "zh": "正例（可选 \u2014 每行一个，格式：标签：示例文本）",
    },
    "qt.s4.positive_ex_placeholder": {
        "en": (
            "W: Increase minimum living allowance subsidies\n"
            "R: Protect citizens' right to know\n"
            "N: The weather is nice today"
        ),
        "zh": (
            "W：加大低保补贴力度\n"
            "R：保障公民知情权\n"
            "N：今天天气不错"
        ),
    },
    "qt.s4.negative_ex": {
        "en": "Negative examples (optional \u2014 commonly misclassified edge cases, same format)",
        "zh": "反例（可选 \u2014 容易误判的边界案例，格式同上）",
    },
    "qt.s4.negative_ex_placeholder": {
        "en": (
            "W: Mentions security but actually discusses legal rights\n"
            "R: Mentions rights but mainly discusses welfare policy"
        ),
        "zh": (
            "W：虽然提到保障但实际讨论的是法律权利\n"
            "R：提到权利但主要讨论福利政策"
        ),
    },
    "qt.s4.generate_prompt": {
        "en": "Generate Prompt (call LLM)",
        "zh": "生成 Prompt（调用 LLM）",
    },
    "qt.s4.current_prompt_title": {
        "en": "### Current Prompt\nThe text box below is the final classification instruction sent to the LLM. You may edit it directly.",
        "zh": "### 当前 Prompt\n下方文本框即为最终发送给 LLM 的分类指令，可直接编辑。",
    },
    "qt.s4.current_prompt": {
        "en": "Current Prompt (editable)",
        "zh": "当前 Prompt（可编辑）",
    },
    "qt.s4.current_prompt_placeholder": {
        "en": (
            "Click \"Generate Prompt\" above to auto-generate via LLM, or enter a custom Prompt here...\n"
            "Please keep the {text} placeholder in your Prompt; it will be replaced with actual text during classification."
        ),
        "zh": (
            "点击上方「生成 Prompt」由 LLM 自动生成，或直接在此输入自定义 Prompt …\n"
            "Prompt 中请保留 {text} 占位符，执行分类时会被替换为实际文本。"
        ),
    },
    "qt.s4.optimize_title": {
        "en": (
            "### Step 2: Optimize & Compare (Optional)\n"
            "First evaluate the Prompt quality to get improvement suggestions, "
            "then manually edit different versions for comparison testing."
        ),
        "zh": (
            "### 第二步：优化 & 对比（可选）\n"
            "先评估 Prompt 质量获取改进建议，再手动编辑不同版本进行对比测试。"
        ),
    },
    "qt.s4.eval_btn": {
        "en": "Evaluate Prompt quality",
        "zh": "评估 Prompt 质量",
    },
    "qt.s4.copy_btn": {
        "en": "Copy to comparison \u2192",
        "zh": "复制到对比栏 \u2192",
    },
    "qt.s4.eval_result": {
        "en": "Evaluation results & improvement suggestions",
        "zh": "评估结果 & 改进建议",
    },
    "qt.s4.compare_desc": {
        "en": "**Comparison versions** (copy the current Prompt and modify for testing different approaches):",
        "zh": "**对比版本**（将当前 Prompt 复制过来修改，测试不同写法的效果）：",
    },
    "qt.s4.variant_1": {
        "en": "Comparison version 1",
        "zh": "对比版本 1",
    },
    "qt.s4.variant_2": {
        "en": "Comparison version 2",
        "zh": "对比版本 2",
    },
    "qt.s4.variant_3": {
        "en": "Comparison version 3",
        "zh": "对比版本 3",
    },
    "qt.s4.use_v1": {
        "en": "Adopt version 1",
        "zh": "采用版本 1",
    },
    "qt.s4.use_v2": {
        "en": "Adopt version 2",
        "zh": "采用版本 2",
    },
    "qt.s4.use_v3": {
        "en": "Adopt version 3",
        "zh": "采用版本 3",
    },
    "qt.s4.test_btn": {
        "en": "Test comparison",
        "zh": "测试对比",
    },
    "qt.s4.test_result": {
        "en": "Test results",
        "zh": "测试结果",
    },
    "qt.s4.test_detail": {
        "en": "Detailed prediction comparison (ground-truth label vs. each variant's prediction per text)",
        "zh": "详细预测对比（每条文本的真实标签 vs 各变体预测）",
    },
    "qt.s4.run_title": {
        "en": (
            "### Step 3: Run Classification\n"
            "After confirming the Prompt, click to run. "
            "**Only unlabeled texts will be classified; texts with existing labels will not be re-classified.**"
        ),
        "zh": (
            "### 第三步：执行分类\n"
            "确认 Prompt 后，点击执行。**只会对未标注的文本进行分类，已有标签的不会重新分类。**"
        ),
    },
    "qt.s4.run_btn": {
        "en": "Start classification",
        "zh": "开始分类",
    },
    "qt.s4.result": {
        "en": "Classification results",
        "zh": "分类结果",
    },
    "qt.s4.detail": {
        "en": "Classification details",
        "zh": "分类详情",
    },

    # -- Fine-tuning sub-tab --

    "qt.s4.ft_tab": {
        "en": "Fine-tuning",
        "zh": "Fine-tuning",
    },
    "qt.s4.ft_explainer": {
        "en": (
            "**Local Fine-tuning Instructions**\n\n"
            "Fine-tuning runs on your local machine without any API calls. Workflow:\n"
            "1. Download a pre-trained model from HuggingFace Hub (approx. 500 MB on first run)\n"
            "2. Fine-tune the model with your labeled data (automatic 80/20 train/validation split)\n"
            "3. Training uses early stopping, selecting the best epoch by Macro-F1\n"
            "4. After training, the model is saved locally at `./socialscikit_model/`\n\n"
            "**Environment requirements:**\n"
            "```\npip install torch transformers datasets\n```\n"
            "- Runs without GPU (automatically uses CPU), but will be slower\n"
            "- Automatically accelerated with NVIDIA GPU + CUDA\n"
            "- Apple Silicon Mac supports MPS acceleration (PyTorch >= 2.0)\n\n"
            "**Recommended data volume:** >= 200 labeled samples (Macro-F1 typically reaches 0.80+); "
            ">= 50 samples can run but with limited effectiveness."
        ),
        "zh": (
            "**本地 Fine-tuning 说明**\n\n"
            "Fine-tuning 在你的本地机器上运行，不经过任何 API。流程：\n"
            "1. 从 HuggingFace Hub 下载预训练模型（首次约 500 MB）\n"
            "2. 用你标注的数据微调模型（自动 80/20 训练/验证集划分）\n"
            "3. 训练过程使用 early stopping，按 Macro-F1 选最优 epoch\n"
            "4. 训练完成后模型保存在本地 `./socialscikit_model/`\n\n"
            "**环境要求：**\n"
            "```\npip install torch transformers datasets\n```\n"
            "- 无 GPU 也可以跑（自动使用 CPU），但会较慢\n"
            "- 有 NVIDIA GPU + CUDA 时自动加速\n"
            "- Apple Silicon Mac 支持 MPS 加速（PyTorch \u2265 2.0）\n\n"
            "**推荐数据量：** \u2265 200 条标注数据（Macro-F1 通常可达 0.80+）；\u2265 50 条可跑但效果有限。"
        ),
    },
    "qt.s4.ft_model": {
        "en": "Model",
        "zh": "模型",
    },
    "qt.s4.ft_batch_size": {
        "en": "Batch Size",
        "zh": "Batch Size",
    },
    "qt.s4.ft_epochs": {
        "en": "Epochs",
        "zh": "Epochs",
    },
    "qt.s4.ft_lr": {
        "en": "Learning Rate",
        "zh": "Learning Rate",
    },
    "qt.s4.ft_start": {
        "en": "Start training",
        "zh": "开始训练",
    },
    "qt.s4.ft_result": {
        "en": "Training results",
        "zh": "训练结果",
    },
    "qt.s4.ft_predictions": {
        "en": "Prediction results",
        "zh": "预测结果",
    },

    # -- API Fine-tuning sub-tab --

    "qt.s4.aft_tab": {
        "en": "API Fine-tuning",
        "zh": "API Fine-tuning",
    },
    "qt.s4.aft_desc": {
        "en": (
            "**Fine-tune a model on OpenAI servers.**\n\n"
            "- Requires an OpenAI API Key (paid account required)\n"
            "- At least 10 labeled samples (50-100+ recommended)\n"
            "- Training takes approx. 10-60 minutes; predictions run automatically on all texts after training\n"
            "- Cost reference: gpt-4o-mini approx. $0.003/1K tokens"
        ),
        "zh": (
            "**在 OpenAI 服务器上微调模型。**\n\n"
            "- 需要 OpenAI API Key（需启用付费账户）\n"
            "- 至少 10 条标注数据（建议 50-100+ 条）\n"
            "- 训练时间约 10-60 分钟，训练完成后自动对全部文本进行预测\n"
            "- 费用参考：gpt-4o-mini 约 $0.003/1K tokens"
        ),
    },
    "qt.s4.aft_key": {
        "en": "OpenAI API Key",
        "zh": "OpenAI API Key",
    },
    "qt.s4.aft_base_model": {
        "en": "Base model",
        "zh": "基础模型",
    },
    "qt.s4.aft_epochs": {
        "en": "Training epochs",
        "zh": "训练轮数",
    },
    "qt.s4.aft_suffix": {
        "en": "Model suffix (optional)",
        "zh": "模型后缀（可选）",
    },
    "qt.s4.aft_submit": {
        "en": "Submit training job",
        "zh": "提交训练任务",
    },
    "qt.s4.aft_status": {
        "en": "Training status",
        "zh": "训练状态",
    },
    "qt.s4.aft_refresh": {
        "en": "Refresh status",
        "zh": "刷新状态",
    },
    "qt.s4.aft_cancel": {
        "en": "Cancel training",
        "zh": "取消训练",
    },
    "qt.s4.aft_predictions": {
        "en": "Prediction results",
        "zh": "预测结果",
    },

    # ======================================================================
    # QuantiKit Step 5 - Evaluation
    # ======================================================================

    "qt.s5.tab": {
        "en": "Step 5 \u00b7 Evaluation",
        "zh": "Step 5 \u00b7 评估",
    },
    "qt.s5.title": {
        "en": (
            "## Evaluation\n"
            "Compare classification results against ground-truth labels using the label column set in Step 1."
        ),
        "zh": (
            "## 评估\n"
            "将分类结果与真实标签对比，使用 Step 1 中设定的标签列。"
        ),
    },
    "qt.s5.run_btn": {
        "en": "Run evaluation",
        "zh": "运行评估",
    },
    "qt.s5.report": {
        "en": "Evaluation report",
        "zh": "评估报告",
    },

    # ======================================================================
    # QuantiKit Step 6 - Export
    # ======================================================================

    "qt.s6.tab": {
        "en": "Step 6 \u00b7 Export",
        "zh": "Step 6 \u00b7 导出",
    },
    "qt.s6.title": {
        "en": "## Export Results\nDownload the classification results as a CSV file.",
        "zh": "## 导出结果\n下载分类结果 CSV 文件。",
    },
    "qt.s6.export_btn": {
        "en": "Export CSV",
        "zh": "导出 CSV",
    },
    "qt.s6.file": {
        "en": "Download",
        "zh": "下载",
    },

    # ======================================================================
    # Annotation statistics format strings
    # ======================================================================

    "msg.ann_stats": {
        "en": "Total: {} | Labeled: {} | Skipped: {} | Flagged: {} | Remaining: {}",
        "zh": "总计：{} 条 | 已标注：{} | 已跳过：{} | 已标记：{} | 剩余：{}",
    },
    "msg.labeled_n": {
        "en": "Labeled: {}",
        "zh": "已标注：{}",
    },
    "msg.skipped_n_already": {
        "en": "Skipped (already labeled): {}",
        "zh": "跳过已有标签：{}",
    },
    "msg.training_submitted": {
        "en": "Training job submitted!",
        "zh": "训练任务已提交！",
    },

    # ======================================================================
    # Common / shared UI labels
    # ======================================================================

    "common.home": {
        "en": "Home",
        "zh": "Home",
    },
    "common.quantikit": {
        "en": "QuantiKit",
        "zh": "QuantiKit",
    },
    "common.qualikit": {
        "en": "QualiKit",
        "zh": "QualiKit",
    },
    "common.backend": {
        "en": "LLM Backend",
        "zh": "LLM 后端",
    },
    "common.model": {
        "en": "Model",
        "zh": "模型",
    },
    "common.api_key": {
        "en": "API Key",
        "zh": "API Key",
    },

    # ======================================================================
    # Inter-Coder Reliability (ICR)
    # ======================================================================

    "icr.title": {
        "en": "Inter-Coder Reliability",
        "zh": "编码者间信度",
    },
    "icr.description": {
        "en": "Compute agreement metrics between two sets of labels or coders.",
        "zh": "计算两组标签或编码者之间的一致性指标。",
    },
    "icr.upload_second_labels": {
        "en": "Upload second coder's labels (CSV)",
        "zh": "上传第二编码者标签（CSV）",
    },
    "icr.second_label_col": {
        "en": "Second coder label column",
        "zh": "第二编码者标签列名",
    },
    "icr.compute_btn": {
        "en": "Compute ICR",
        "zh": "计算编码者间信度",
    },
    "icr.report": {
        "en": "ICR Report",
        "zh": "信度报告",
    },
    "icr.human_vs_llm": {
        "en": "Human vs LLM Agreement",
        "zh": "人工 vs LLM 一致性",
    },
    "icr.human_vs_llm_desc": {
        "en": "Compare human-reviewed themes against original LLM coding.",
        "zh": "对比人工审核主题与原始 LLM 编码结果。",
    },
    "icr.compute_human_llm_btn": {
        "en": "Compute Human vs LLM ICR",
        "zh": "计算人工 vs LLM 信度",
    },
    "msg.run_eval_first": {
        "en": "Please run evaluation first.",
        "zh": "请先运行评估。",
    },
    "msg.no_review_data": {
        "en": "No review data available. Please complete coding and review first.",
        "zh": "暂无审核数据，请先完成编码和审核。",
    },

    # ======================================================================
    # Multi-LLM Consensus Coding
    # ======================================================================

    "consensus.title": {
        "en": "Consensus Coding (Multi-LLM)",
        "zh": "共识编码（多模型）",
    },
    "consensus.description": {
        "en": (
            "Run 2–3 LLMs independently on the same segments. "
            "Themes are retained only when a majority of models agree."
        ),
        "zh": (
            "使用 2–3 个 LLM 分别独立编码同一批文本，"
            "仅保留多数模型一致同意的主题。"
        ),
    },
    "consensus.backend_n": {
        "en": "LLM {} Backend",
        "zh": "LLM {} 后端",
    },
    "consensus.model_n": {
        "en": "Model {}",
        "zh": "模型 {}",
    },
    "consensus.api_key_n": {
        "en": "API Key {}",
        "zh": "API Key {}",
    },
    "consensus.run_btn": {
        "en": "Run Consensus Coding",
        "zh": "运行共识编码",
    },
    "consensus.summary": {
        "en": "Consensus Summary",
        "zh": "共识摘要",
    },
    "consensus.results": {
        "en": "Consensus Results",
        "zh": "共识结果",
    },
    "consensus.agreement": {
        "en": "Agreement Report",
        "zh": "一致性报告",
    },
    "msg.at_least_two_llms": {
        "en": "Please configure at least 2 LLMs for consensus coding.",
        "zh": "共识编码需要至少配置 2 个 LLM。",
    },
    "msg.consensus_done": {
        "en": "Consensus coding complete. {} segments coded with {} models.",
        "zh": "共识编码完成。{} 条文本使用 {} 个模型编码。",
    },
    "msg.lock_themes_first": {
        "en": "Please define and lock themes first.",
        "zh": "请先定义并锁定主题框架。",
    },

    # ======================================================================
    # Methods Section Auto-generation
    # ======================================================================

    "methods.title": {
        "en": "Methods Section Generator",
        "zh": "方法论段落生成",
    },
    "methods.description": {
        "en": "Generate a Methods paragraph draft for your paper based on the analysis pipeline.",
        "zh": "根据分析流程自动生成论文方法论段落草稿。",
    },
    "methods.generate_btn": {
        "en": "Generate Methods Section",
        "zh": "生成方法论段落",
    },
    "methods.text_en": {
        "en": "Methods (English)",
        "zh": "方法论（英文）",
    },
    "methods.text_zh": {
        "en": "Methods (Chinese)",
        "zh": "方法论（中文）",
    },
    "methods.copy_hint": {
        "en": "Auto-generated draft. Copy, edit, and cite appropriately before publication.",
        "zh": "自动生成草稿，请复制后编辑，并在发表前适当引用。",
    },
    "methods.no_data": {
        "en": "Please complete the analysis pipeline before generating.",
        "zh": "请先完成分析流程再生成。",
    },

    # ======================================================================
    # Toolbox
    # ======================================================================

    "toolbox.title": {
        "en": "Toolbox",
        "zh": "工具箱",
    },
    "toolbox.description": {
        "en": "Standalone research tools — ICR Calculator, Multi-LLM Consensus Coding, and Methods Section Generator. These tools work independently from QuantiKit and QualiKit.",
        "zh": "独立研究工具 — 编码者间信度计算、多模型共识编码、方法论段落生成。这些工具独立于 QuantiKit 和 QualiKit 使用。",
    },
    "toolbox.icr_tab": {
        "en": "ICR Calculator",
        "zh": "编码者间信度",
    },
    "toolbox.consensus_tab": {
        "en": "Consensus Coding",
        "zh": "共识编码",
    },
    "toolbox.methods_tab": {
        "en": "Methods Generator",
        "zh": "方法论生成",
    },
    "toolbox.import_log": {
        "en": "Import Pipeline Log",
        "zh": "导入流水线日志",
    },
    "toolbox.export_log": {
        "en": "Export Pipeline Log",
        "zh": "导出流水线日志",
    },
    "toolbox.manual_input": {
        "en": "Manual Input (without log file)",
        "zh": "手动输入（无日志文件时使用）",
    },
    "toolbox.pipeline_type": {
        "en": "Pipeline Type",
        "zh": "流水线类型",
    },
    "toolbox.icr_upload": {
        "en": "Upload Labels CSV (each column = one coder)",
        "zh": "上传标签 CSV（每列代表一个编码者）",
    },
    "toolbox.icr_file_info": {
        "en": "File info",
        "zh": "文件信息",
    },
    "toolbox.icr_select_cols": {
        "en": "Select coder columns",
        "zh": "选择编码者列",
    },
    "toolbox.icr_select_cols_info": {
        "en": "Pick 2+ columns. 2 coders → Cohen's Kappa; 3+ coders → Krippendorff's Alpha.",
        "zh": "选择 2 列以上。2 人 → Cohen's Kappa；3 人以上 → Krippendorff's Alpha。",
    },
    "toolbox.icr_mode": {
        "en": "Label Mode",
        "zh": "标签模式",
    },
    "toolbox.icr_mode_info": {
        "en": "single-label: one label per cell. multi-label: comma-separated values per cell.",
        "zh": "单标签：每格一个标签。多标签：每格用逗号分隔多个标签。",
    },
    "toolbox.add_llm": {
        "en": "+ Add LLM",
        "zh": "+ 添加模型",
    },
    "toolbox.remove_llm": {
        "en": "- Remove LLM",
        "zh": "- 移除模型",
    },
    "toolbox.data_file": {
        "en": "Data File (CSV)",
        "zh": "数据文件（CSV）",
    },
    "toolbox.text_col": {
        "en": "Text Column",
        "zh": "文本列名",
    },
    "toolbox.themes_input": {
        "en": "Themes (one per line, format: name: description)",
        "zh": "主题（每行一个，格式：名称: 描述）",
    },
    "toolbox.download_example": {
        "en": "Download Example",
        "zh": "下载示例",
    },
    "toolbox.example_file": {
        "en": "Example File",
        "zh": "示例文件",
    },
    "toolbox.example_qt_log": {
        "en": "QuantiKit Log Example",
        "zh": "QuantiKit 日志示例",
    },
    "toolbox.example_ql_log": {
        "en": "QualiKit Log Example",
        "zh": "QualiKit 日志示例",
    },
}
