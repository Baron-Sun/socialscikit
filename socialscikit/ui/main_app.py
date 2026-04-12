"""Unified Gradio Web UI for SocialSciKit.

Landing page → QuantiKit (text classification) / QualiKit (qualitative coding).
Design: flat, academic, minimal.
"""

from __future__ import annotations

import json
import os
import tempfile

import gradio as gr
import pandas as pd

from socialscikit.core.data_loader import get_template_path
from socialscikit.quantikit.feature_extractor import TASK_TYPES

import socialscikit.ui.quantikit_app as qn
import socialscikit.ui.qualikit_app as ql
import socialscikit.ui.toolbox_app as tb
from socialscikit.ui.i18n import t, LANGUAGES

# ---------------------------------------------------------------------------
# JS — force light mode (prevent Gradio from following system dark theme)
# ---------------------------------------------------------------------------

_FORCE_LIGHT_JS = """
() => {
    /* ---- Force light mode ---- */
    const forceLightMode = () => {
        document.documentElement.classList.remove('dark');
        document.body.classList.remove('dark');
    };
    forceLightMode();
    new MutationObserver(forceLightMode).observe(document.body, {
        attributes: true, attributeFilter: ['class']
    });
    new MutationObserver(forceLightMode).observe(document.documentElement, {
        attributes: true, attributeFilter: ['class']
    });

    /* ---- Helper: current language ---- */
    const getLang = () => {
        const r = document.querySelector('#lang_radio input[type="radio"]:checked');
        return (r && r.value === '中文') ? 'zh' : 'en';
    };

    /* ---- Fix file-upload drop-zone text (bare TEXT NODEs) ---- */
    const fixDropText = () => {
        const zh = getLang() === 'zh';
        document.querySelectorAll('button.center .wrap, button.flex .wrap').forEach(w => {
            w.childNodes.forEach(n => {
                if (n.nodeType !== 3) return;
                const t = n.textContent.trim();
                if (!zh) {
                    if (t.includes('拖放') || t === '将文件拖放到此处')
                        n.textContent = '\\nDrop file here\\n';
                    else if (t === '点击上传')
                        n.textContent = '\\nClick to upload\\n';
                } else {
                    if (t === 'Drop file here')
                        n.textContent = '\\n将文件拖放到此处\\n';
                    else if (t === 'Click to upload')
                        n.textContent = '\\n点击上传\\n';
                }
            });
            const or = w.querySelector('.or');
            if (or) {
                if (!zh && or.textContent.includes('或')) or.textContent = '- or -';
                else if (zh && or.textContent.includes('or')) or.textContent = '- 或 -';
            }
        });
    };

    /* ---- Tab label i18n (JS-driven, since Gradio tabs are static) ---- */
    const TAB_I18N = {
        'Home':                       '首页',
        'Step 1 · Data Upload':       '步骤 1 · 数据上传',
        'Step 2 · Recommendation':    '步骤 2 · 方法推荐',
        'Step 3 · Annotation':        '步骤 3 · 数据标注',
        'Step 4 · Classification':    '步骤 4 · 文本分类',
        'Step 5 · Evaluation':        '步骤 5 · 模型评估',
        'Step 6 · Export':            '步骤 6 · 导出',
        'Prompt Classification':      '提示词分类',
        'Fine-tuning':                '本地微调',
        'API Fine-tuning':            'API 微调',
        'Step 1 · Upload & Segment':  '步骤 1 · 上传与分段',
        'Step 2 · De-identification': '步骤 2 · 脱敏处理',
        'Step 3 · Research Framework':'步骤 3 · 研究框架',
        'Step 4 · LLM Coding':        '步骤 4 · LLM 编码',
        'Step 5 · Review':            '步骤 5 · 人工审核',
    };
    const TAB_REV = {};
    for (const [en, zh] of Object.entries(TAB_I18N)) TAB_REV[zh] = en;

    const switchTabs = () => {
        const zh = getLang() === 'zh';
        document.querySelectorAll('.tab-nav button').forEach(btn => {
            const t = btn.textContent.trim();
            if (zh && TAB_I18N[t])  btn.textContent = TAB_I18N[t];
            if (!zh && TAB_REV[t])  btn.textContent = TAB_REV[t];
        });
    };

    /* ---- Wire language-change listener ---- */
    const onLangChange = () => { switchTabs(); fixDropText(); };

    /* Listen on radio inputs */
    const lr = document.getElementById('lang_radio');
    if (lr) {
        lr.addEventListener('change', () => setTimeout(onLangChange, 120));
        /* Also observe attribute changes (aria-checked) as fallback */
        new MutationObserver(() => setTimeout(onLangChange, 120))
            .observe(lr, { subtree: true, attributes: true });
    }

    /* Initial fix for file-drop text (browser locale → Chinese) */
    fixDropText();
    new MutationObserver(fixDropText).observe(document.body, {
        childList: true, subtree: true
    });
    /* Fallback for stubborn Gradio re-renders */
    setInterval(fixDropText, 3000);
}
"""

# ---------------------------------------------------------------------------
# CSS — academic flat style
# ---------------------------------------------------------------------------

_CSS = """
/* ============================================================
   SocialSciKit — light flat academic theme
   白底 · 深色文字 · 无多余边框 · 扁平
   ============================================================ */

/* ---- Force light mode — override Gradio dark mode at CSS variable level ---- */
:root, :host {
    color-scheme: light only !important;
}
/* When Gradio adds .dark class, force all variables back to light values */
.dark,
body.dark,
html.dark,
:root.dark,
.dark :root,
.dark :host {
    --body-background-fill: #ffffff !important;
    --body-text-color: #333333 !important;
    --body-text-color-subdued: #777777 !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #fafafa !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #f0f0f0 !important;
    --block-label-text-color: #444444 !important;
    --block-title-text-color: #333333 !important;
    --panel-background-fill: #ffffff !important;
    --panel-border-color: #f0f0f0 !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #e5e7eb !important;
    --input-placeholder-color: #aaaaaa !important;
    --border-color-primary: #e5e7eb !important;
    --border-color-accent: #4A90D9 !important;
    --neutral-50: #fafafa !important;
    --neutral-100: #f5f5f5 !important;
    --neutral-200: #e5e5e5 !important;
    --neutral-300: #d4d4d4 !important;
    --neutral-400: #a3a3a3 !important;
    --neutral-500: #737373 !important;
    --neutral-600: #525252 !important;
    --neutral-700: #404040 !important;
    --neutral-800: #262626 !important;
    --neutral-900: #171717 !important;
    --neutral-950: #0a0a0a !important;
    --color-accent: #333 !important;
    --color-accent-soft: #f5f5f5 !important;
    --text-color: #333333 !important;
    --text-color-subdued: #777777 !important;
    --shadow-drop: none !important;
    --shadow-drop-lg: none !important;
    --button-secondary-background-fill: #ffffff !important;
    --button-secondary-text-color: #555555 !important;
    --button-secondary-border-color: #d1d5db !important;
    --table-even-background-fill: #ffffff !important;
    --table-odd-background-fill: #fafafa !important;
    --table-row-focus: #f5f5f5 !important;
    --loader-color: #4A90D9 !important;
    --error-background-fill: #fef2f2 !important;
    --error-border-color: #fca5a5 !important;
    --error-text-color: #991b1b !important;
    color-scheme: light only !important;
}
/* Belt-and-suspenders: force element colors under .dark */
.dark .gradio-container,
.dark .block, .dark .panel, .dark .form, .dark .tabitem {
    background: #ffffff !important;
    color: #333333 !important;
}
.dark textarea, .dark input[type="text"], .dark input[type="password"],
.dark input[type="number"], .dark select {
    background: #ffffff !important;
    color: #333333 !important;
    border-color: #e0e0e0 !important;
}
.dark textarea::placeholder, .dark input::placeholder {
    color: #aaaaaa !important;
}
.dark label span, .dark .prose, .dark .prose * {
    color: inherit !important;
}
.dark .table-wrap, .dark .table-wrap th, .dark .table-wrap td {
    background: inherit !important;
    color: #333333 !important;
}
.dark .tabs > .tab-nav > button { color: #aaa !important; background: transparent !important; }
.dark .tabs > .tab-nav > button.selected { color: #333 !important; }

/* ---- Global color variables override ---- */
:root,
:host {
    --color-accent: #333 !important;
    --color-accent-soft: #f5f5f5 !important;
}

/* ---- Base ---- */
.gradio-container {
    max-width: 1080px !important;
    margin: auto !important;
    font-family: "Inter", -apple-system, "Helvetica Neue", Arial, sans-serif !important;
    background: #fff !important;
    color: #333 !important;
}

/* ---- Kill ALL borders, shadows, backgrounds on wrappers ---- */
.block, .panel, .form, .group,
.tabitem, .tab-content,
.block.padded, .block.border,
div[class*="block"], div[class*="panel"] {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* ---- Top-level tabs — minimal underline ---- */
.tabs > .tab-nav {
    border-bottom: 1px solid #eee !important;
    gap: 0 !important;
    background: transparent !important;
}
.tabs > .tab-nav > button {
    font-weight: 500 !important;
    font-size: 0.93rem !important;
    padding: 10px 22px !important;
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: #aaa !important;
}
.tabs > .tab-nav > button:hover { color: #666 !important; }
.tabs > .tab-nav > button.selected {
    color: #333 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #555 !important;
    background: transparent !important;
}

/* ---- Typography — all text stays dark & readable ---- */
.prose h1 { font-size: 1.5rem !important; font-weight: 700 !important; color: #222 !important; }
.prose h2 { font-size: 1.1rem !important; font-weight: 600 !important; color: #333 !important; margin-top: 0.5rem !important; }
.prose h3 { font-size: 1.0rem !important; font-weight: 600 !important; color: #333 !important; }
.prose p, .prose li { font-size: 0.92rem !important; color: #444 !important; line-height: 1.7 !important; }

/* Force bold & links to be dark — override Gradio's accent color */
.prose strong, .prose b { color: #222 !important; }
.prose a { color: #333 !important; text-decoration: underline !important; }
.prose a:hover { color: #000 !important; }

.prose table { font-size: 0.88rem !important; }
.prose table th { background: #fafafa !important; color: #333 !important; font-weight: 600 !important; }
.prose table td { color: #444 !important; }
.prose blockquote { border-left: 3px solid #ddd !important; color: #666 !important; font-size: 0.88rem !important; background: transparent !important; }
.prose hr { border-color: #eee !important; }
.prose code { background: #f5f5f5 !important; color: #333 !important; font-size: 0.85rem !important; padding: 2px 6px; border-radius: 3px; }

/* summary / details */
.prose summary { color: #333 !important; font-weight: 600 !important; }

/* ---- Buttons ---- */
.primary {
    background: #4A90D9 !important;
    border: none !important;
    color: #fff !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
}
.primary:hover { background: #3A7BC8 !important; }

.secondary {
    background: #fff !important;
    border: 1px solid #ddd !important;
    color: #444 !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
}
.secondary:hover { background: #fafafa !important; }

/* ---- Input fields ---- */
textarea, input[type="text"], input[type="password"], input[type="number"] {
    border: 1px solid #e0e0e0 !important;
    border-radius: 6px !important;
    background: #fff !important;
    color: #333 !important;
    font-size: 0.9rem !important;
}
textarea:focus, input:focus {
    border-color: #999 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* ---- Labels ---- */
label span { font-weight: 500 !important; color: #333 !important; font-size: 0.88rem !important; }

/* ---- Dataframe ---- */
.table-wrap { border: none !important; box-shadow: none !important; }
.table-wrap table { font-size: 0.85rem !important; color: #333 !important; }
.table-wrap th { background: #fafafa !important; color: #333 !important; font-weight: 600 !important; }
.table-wrap td { color: #444 !important; }

/* ---- Accordion ---- */
.accordion { border: none !important; box-shadow: none !important; background: transparent !important; }
.accordion .label-wrap { font-weight: 500 !important; color: #333 !important; }

/* ---- Dropdowns ---- */
.wrap[data-testid="dropdown"], select {
    background: #fff !important;
    color: #333 !important;
    border: 1px solid #e0e0e0 !important;
}

/* ---- Checkbox ---- */
.checkbox-container label { color: #333 !important; }

/* ---- Textbox (read-only) ---- */
.textbox textarea[disabled], textarea[readonly] {
    background: #fafafa !important;
    color: #333 !important;
    border-color: #eee !important;
}

/* ---- File upload ---- */
.upload-container { border: 1px dashed #ccc !important; background: #fafafa !important; }

/* ---- Slider ---- */
.range-slider input[type="range"] { accent-color: #4A90D9 !important; }

/* ---- Number input ---- */
.number-input input { color: #333 !important; background: #fff !important; }

/* ---- Progress bar / loading indicator ---- */
.progress-bar { background: #4A90D9 !important; }
.progress-bar-wrap {
    background: #e5e7eb !important;
    border: 1px solid #d1d5db !important;
}
.eta-bar {
    background: rgba(74, 144, 217, 0.20) !important;
    opacity: 1 !important;
}
.progress-text { color: #333 !important; }
.progress-level { color: #333 !important; }
.progress-level-inner { color: #333 !important; }
.meta-text, .meta-text-center { color: #666 !important; }
/* Generating pulse border — make it visible */
.wrap.generating { border-color: #4A90D9 !important; }
/* Dark mode overrides for progress */
.dark .progress-bar { background: #4A90D9 !important; }
.dark .progress-bar-wrap { background: #e5e7eb !important; border-color: #d1d5db !important; }
.dark .eta-bar { background: rgba(74, 144, 217, 0.20) !important; opacity: 1 !important; }
.dark .progress-text, .dark .progress-level, .dark .progress-level-inner { color: #333 !important; }
.dark .meta-text, .dark .meta-text-center { color: #666 !important; }
.dark .wrap.generating { border-color: #4A90D9 !important; }
/* Toast notifications */
.toast-body { background: #fff !important; color: #333 !important; border: 1px solid #e5e7eb !important; }
.toast-body.info { border-left: 4px solid #4A90D9 !important; }
.toast-title { color: #333 !important; }
.toast-text p { color: #444 !important; }
.dark .toast-body { background: #fff !important; color: #333 !important; }
.dark .toast-title { color: #333 !important; }
.dark .toast-text p { color: #444 !important; }

/* ---- Language selector — inline radio ---- */
#lang_radio { max-width: 300px !important; min-width: 250px !important; }
#lang_radio .wrap {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 16px !important;
    align-items: center !important;
}
#lang_radio label { white-space: nowrap !important; }

/* ---- Footer hide ---- */
footer { display: none !important; }

/* ---- Landing page ---- */
.hero-title { text-align: center; padding: 2.5rem 0 0.5rem; }
.hero-title h1 { font-size: 2rem !important; color: #222 !important; margin-bottom: 0.25rem !important; }
.hero-sub { text-align: center; color: #888 !important; font-size: 1rem !important; margin-bottom: 2rem; }

/* ---- Module cards — just a subtle bg, no border ---- */
.module-card {
    border: none !important;
    border-radius: 8px;
    padding: 1.5rem;
    background: #f7f7f8;
}
.module-card h3 { margin-top: 0 !important; color: #222 !important; }
.module-card strong { color: #222 !important; }
.module-card p, .module-card li { color: #444 !important; }
"""

# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

_LANDING = """\
<div class="hero-title">

# SocialSciKit

</div>
<div class="hero-sub">

Zero-code text analysis toolkit for social science researchers

</div>

---

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin: 1.5rem 0;">
<div class="module-card">

### QuantiKit — 文本分类

**适用场景：** 情感分析、立场检测、框架分析、主题分类

**流程：** 上传数据 → 方法推荐 → 标注 → 分类 → 评估 → 导出

**能力：**
- 方法推荐引擎（附 CSS 文献依据）
- 标注预算推荐（学习曲线拟合）
- Zero-shot / Few-shot / Fine-tuning
- Prompt 多版本自动优化

</div>
<div class="module-card">

### QualiKit — 质性编码

**适用场景：** 访谈分析、开放式问卷、文献主题编码

**流程：** 上传数据 → 脱敏 → 主题定义 → 编码 → 导出

**能力：**
- PII 自动检测与交互审核
- AI 辅助主题建议（TF-IDF / LLM）
- LLM 多标签编码 + 置信度分级
- 摘录表 + 共现矩阵 + 分析备忘录

</div>
</div>

### 快速开始

1. 点击上方 **QuantiKit** 或 **QualiKit** 标签页进入对应模块
2. 按步骤编号依次操作 — 每步结果可审核编辑，确认后再进入下一步
3. 需要 LLM 功能时提供 API Key（OpenAI / Anthropic），或使用 Ollama 本地推理

**示例数据：**
`examples/sentiment_example.csv`（情感分类 · QuantiKit）·
`examples/policy_example.csv`（政策工具分类 · QuantiKit）·
`examples/interview_example.txt`（单人访谈 · QualiKit）·
`examples/interview_focus_group.txt`（焦点小组 · QualiKit）

<details>
<summary style="cursor:pointer; font-weight:600; color:#2c3e50;">参考文献</summary>

- Ziems, C. et al. (2024). Can LLMs transform computational social science? *Computational Linguistics*, 50(1).
- Chae, Y. & Davidson, T. (2025). LLMs for text classification. *Sociological Methods & Research*.
- Dunivin, Z. O. (2024). Scalable qualitative coding with LLMs. *arXiv:2401.15170*.
- Do, S. et al. (2024). The augmented social scientist. *SMR*, 53(3).
- Zhou, Y. et al. (2022). Large Language Models Are Human-Level Prompt Engineers. *ICLR 2023*.

</details>
"""

# ---------------------------------------------------------------------------
# Fine-tuning explainer (for the Fine-tuning sub-tab)
# ---------------------------------------------------------------------------

_FT_EXPLAINER = """\
**本地 Fine-tuning 说明**

Fine-tuning 在你的本地机器上运行，不经过任何 API。流程：
1. 从 HuggingFace Hub 下载预训练模型（首次约 500 MB）
2. 用你标注的数据微调模型（自动 80/20 训练/验证集划分）
3. 训练过程使用 early stopping，按 Macro-F1 选最优 epoch
4. 训练完成后模型保存在本地 `./socialscikit_model/`

**环境要求：**
```
pip install torch transformers datasets
```
- 无 GPU 也可以跑（自动使用 CPU），但会较慢
- 有 NVIDIA GPU + CUDA 时自动加速
- Apple Silicon Mac 支持 MPS 加速（PyTorch ≥ 2.0）

**推荐数据量：** ≥ 200 条标注数据（Macro-F1 通常可达 0.80+）；≥ 50 条可跑但效果有限。
"""


# ---------------------------------------------------------------------------
# i18n helpers
# ---------------------------------------------------------------------------


def _build_landing(lang: str = "en") -> str:
    """Build the landing page HTML using translated strings."""
    return (
        '<div class="hero-title">\n\n'
        f'{t("landing.title_html", lang)}\n\n'
        '</div>\n'
        '<div class="hero-sub">\n\n'
        f'{t("landing.subtitle", lang)}\n\n'
        '</div>\n\n---\n\n'
        '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin: 1.5rem 0;">\n'
        '<div class="module-card">\n\n'
        f'{t("landing.quantikit_card", lang)}\n\n'
        '</div>\n'
        '<div class="module-card">\n\n'
        f'{t("landing.qualikit_card", lang)}\n\n'
        '</div>\n</div>\n\n'
        f'{t("landing.quickstart", lang)}\n\n'
        f'{t("landing.examples", lang)}\n\n'
        f'{t("landing.references", lang)}'
    )


# ---------------------------------------------------------------------------
# Build unified app
# ---------------------------------------------------------------------------


def create_app() -> gr.Blocks:
    """Create the unified SocialSciKit Gradio application."""

    with gr.Blocks(
        title="SocialSciKit",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.gray,
            neutral_hue=gr.themes.colors.gray,
            font=["Inter", "-apple-system", "Helvetica Neue", "Arial", "sans-serif"],
            font_mono=["JetBrains Mono", "Fira Code", "Consolas", "monospace"],
            radius_size=gr.themes.sizes.radius_md,
        ).set(
            # Light mode values
            body_background_fill="#ffffff",
            body_text_color="#333333",
            body_text_color_subdued="#777777",
            block_background_fill="#ffffff",
            block_border_width="0px",
            block_border_color="#f0f0f0",
            block_shadow="none",
            block_label_text_color="#444444",
            block_title_text_color="#333333",
            panel_background_fill="#ffffff",
            panel_border_width="0px",
            panel_border_color="#f0f0f0",
            input_background_fill="#ffffff",
            input_border_color="#e5e7eb",
            input_border_color_focus="#4A90D9",
            input_shadow_focus="none",
            input_placeholder_color="#aaaaaa",
            button_primary_background_fill="#4A90D9",
            button_primary_background_fill_hover="#3A7BC8",
            button_primary_text_color="#ffffff",
            button_primary_border_color="#4A90D9",
            button_secondary_background_fill="#ffffff",
            button_secondary_background_fill_hover="#f9fafb",
            button_secondary_border_color="#d1d5db",
            button_secondary_text_color="#555555",
            border_color_primary="#e5e7eb",
            border_color_accent="#4A90D9",
            # Dark mode — force identical to light (lock light theme)
            body_background_fill_dark="#ffffff",
            body_text_color_dark="#333333",
            body_text_color_subdued_dark="#777777",
            block_background_fill_dark="#ffffff",
            block_border_width_dark="0px",
            block_border_color_dark="#f0f0f0",
            block_shadow_dark="none",
            block_label_text_color_dark="#444444",
            block_title_text_color_dark="#333333",
            panel_background_fill_dark="#ffffff",
            panel_border_width_dark="0px",
            panel_border_color_dark="#f0f0f0",
            input_background_fill_dark="#ffffff",
            input_border_color_dark="#e5e7eb",
            input_border_color_focus_dark="#4A90D9",
            input_shadow_focus_dark="none",
            input_placeholder_color_dark="#aaaaaa",
            button_primary_background_fill_dark="#4A90D9",
            button_primary_background_fill_hover_dark="#3A7BC8",
            button_primary_text_color_dark="#ffffff",
            button_primary_border_color_dark="#4A90D9",
            button_secondary_background_fill_dark="#ffffff",
            button_secondary_background_fill_hover_dark="#f9fafb",
            button_secondary_border_color_dark="#d1d5db",
            button_secondary_text_color_dark="#555555",
            border_color_primary_dark="#e5e7eb",
            border_color_accent_dark="#4A90D9",
        ),
        css=_CSS,
        js=_FORCE_LIGHT_JS,
    ) as app:

        # ==================================================================
        # Language selector
        # ==================================================================
        with gr.Row():
            lang_selector = gr.Radio(
                choices=["English", "中文"],
                value="English",
                label="Language",
                interactive=True,
                scale=0,
                min_width=260,
                elem_id="lang_radio",
            )
        ql_lang = gr.State("en")

        # ==================================================================
        # Landing
        # ==================================================================
        with gr.Tab("Home"):
            landing_md = gr.Markdown(value=_build_landing("en"))

        # ==================================================================
        # QuantiKit
        # ==================================================================
        with gr.Tab("QuantiKit"):

            qt_df = gr.State(None)
            qt_result_df = gr.State(None)
            qt_ann_session = gr.State(None)

            # -- 1. Upload ------------------------------------------------
            with gr.Tab("Step 1 · Data Upload"):
                qt_s1_md = gr.Markdown(t("qt.s1.title", "en"))
                with gr.Row():
                    qt_file = gr.File(label=t("qt.s1.file", "en"), file_types=[".csv", ".xlsx", ".xls", ".json", ".jsonl", ".txt"])
                    with gr.Column(scale=0, min_width=140):
                        qt_tpl_btn = gr.Button(t("qt.s1.download_template", "en"), variant="secondary", size="sm")
                        qt_tpl_file = gr.File(label=t("qt.s1.template", "en"), visible=False)
                qt_s1_col_md = gr.Markdown(t("qt.s1.col_mapping_title", "en"))
                with gr.Row():
                    qt_tcol = gr.Dropdown(label=t("qt.s1.text_col", "en"), choices=[], interactive=True)
                    qt_lcol = gr.Dropdown(label=t("qt.s1.label_col", "en"), choices=[], interactive=True)
                    qt_col_btn = gr.Button(t("qt.s1.confirm_col", "en"), variant="primary")
                qt_summary = gr.Textbox(label=t("qt.s1.report", "en"), lines=12, interactive=False)
                with gr.Row():
                    qt_issues = gr.Dataframe(label=t("qt.s1.issues", "en"), interactive=False)
                qt_preview = gr.Dataframe(label=t("qt.s1.preview", "en"), interactive=False)
                with gr.Row():
                    qt_fix_btn = gr.Button(t("qt.s1.fix_btn", "en"), variant="secondary")
                    qt_fix_msg = gr.Textbox(label="", interactive=False, show_label=False)

                qt_file.change(fn=qn._load_and_select_columns, inputs=[qt_file],
                               outputs=[qt_df, qt_summary, qt_issues, qt_preview, qt_tcol, qt_lcol])
                qt_col_btn.click(fn=qn._confirm_columns,
                                 inputs=[qt_df, qt_tcol, qt_lcol],
                                 outputs=[qt_summary, qt_issues, qt_preview])
                qt_fix_btn.click(fn=qn._apply_fixes, inputs=[qt_df], outputs=[qt_df, qt_fix_msg])
                qt_tpl_btn.click(fn=qn._download_template, outputs=[qt_tpl_file])

            # -- 2. Recommendation ----------------------------------------
            with gr.Tab("Step 2 · Recommendation"):
                qt_s2_md = gr.Markdown(t("qt.s2.title", "en"))
                with gr.Row():
                    qt_task = gr.Dropdown(choices=sorted(TASK_TYPES), value="sentiment", label=t("qt.s2.task_type", "en"))
                    qt_ncls = gr.Number(label=t("qt.s2.num_classes", "en"), value=2, precision=0)
                    qt_f1 = gr.Slider(0.5, 1.0, value=0.80, step=0.05, label=t("qt.s2.target_f1", "en"))
                    qt_budget = gr.Dropdown(choices=["low", "medium", "high"], value="medium", label=t("qt.s2.budget", "en"))
                qt_rec_btn = gr.Button(t("qt.s2.recommend_btn", "en"), variant="primary")
                with gr.Row():
                    with gr.Column():
                        qt_feat = gr.Textbox(label=t("qt.s2.features", "en"), lines=8, interactive=False)
                    with gr.Column():
                        qt_bgt = gr.Textbox(label=t("qt.s2.annotation_budget", "en"), lines=8, interactive=False)
                qt_rec = gr.Textbox(label=t("qt.s2.recommendation", "en"), lines=12, interactive=False)
                with gr.Row():
                    qt_curve_plot = gr.Plot(label=t("qt.s2.curve_plot", "en"), scale=3)
                    qt_curve_tbl = gr.Dataframe(label=t("qt.s2.key_points", "en"), interactive=False, scale=1)

                qt_rec_btn.click(fn=qn._extract_and_recommend,
                                 inputs=[qt_df, qt_task, qt_ncls, qt_f1, qt_budget, qt_tcol, qt_lcol],
                                 outputs=[qt_feat, qt_rec, qt_bgt, qt_curve_plot, qt_curve_tbl])

            # -- 3. Annotation --------------------------------------------
            with gr.Tab("Step 3 · Annotation"):
                qt_s3_md = gr.Markdown(t("qt.s3.title", "en"))
                with gr.Row():
                    qa_labels = gr.Textbox(label=t("qt.s3.labels", "en"), value="positive, negative, neutral", placeholder=t("qt.s3.labels_placeholder", "en"))
                    qa_shuf = gr.Checkbox(label=t("qt.s3.shuffle", "en"), value=False)
                qa_create = gr.Button(t("qt.s3.create_session", "en"), variant="primary")
                qa_stats = gr.Textbox(label=t("qt.s3.progress", "en"), interactive=False)
                qa_idx = gr.Textbox(label=t("qt.s3.current_pos", "en"), interactive=False)
                qa_text = gr.Textbox(label=t("qt.s3.text_to_annotate", "en"), lines=4, interactive=False)
                with gr.Row():
                    qa_input = gr.Dropdown(
                        label=t("qt.s3.label", "en"), scale=3, allow_custom_value=True,
                        choices=["positive", "negative", "neutral"], interactive=True,
                    )
                    qa_sub = gr.Button(t("qt.s3.annotate_btn", "en"), variant="primary", scale=1)
                    qa_skip = gr.Button(t("qt.s3.skip_btn", "en"), variant="secondary", scale=1)
                with gr.Row():
                    qa_fnote = gr.Textbox(label=t("qt.s3.note", "en"), placeholder=t("qt.s3.note_placeholder", "en"), scale=3)
                    qa_flag = gr.Button(t("qt.s3.flag_btn", "en"), variant="secondary", scale=1)
                    qa_undo = gr.Button(t("qt.s3.undo_btn", "en"), variant="secondary", scale=1)
                qa_msg = gr.Textbox(label="", interactive=False, show_label=False)
                with gr.Accordion(t("qt.s3.export_accordion", "en"), open=False):
                    with gr.Row():
                        qa_all = gr.Checkbox(label=t("qt.s3.include_skipped", "en"), value=False)
                        qa_exp = gr.Button(t("qt.s3.preview_btn", "en"), variant="secondary")
                        qa_dl = gr.Button(t("qt.s3.download_csv", "en"), variant="secondary")
                        qa_merge = gr.Button(t("qt.s3.merge_btn", "en"), variant="secondary")
                    qa_tbl = gr.Dataframe(label=t("qt.s3.results", "en"), interactive=False)
                    qa_dlf = gr.File(label=t("qt.s3.download_file", "en"))
                    qa_mmsg = gr.Textbox(label="", interactive=False, show_label=False)

                _so = [qt_ann_session, qa_stats, qa_text, qa_idx, qa_msg]
                qa_create.click(fn=qn._create_annotation_session,
                                inputs=[qt_df, qt_tcol, qt_lcol, qa_labels, qa_shuf], outputs=_so)
                qa_sub.click(fn=qn._annotate_item, inputs=[qt_ann_session, qa_input], outputs=_so)
                qa_skip.click(fn=qn._skip_item, inputs=[qt_ann_session], outputs=_so)
                qa_flag.click(fn=qn._flag_item, inputs=[qt_ann_session, qa_fnote], outputs=_so)
                qa_undo.click(fn=qn._undo_annotation, inputs=[qt_ann_session], outputs=_so)
                qa_exp.click(fn=qn._export_annotations, inputs=[qt_ann_session, qa_all], outputs=[qa_tbl, qa_msg])
                qa_dl.click(fn=qn._download_annotations_csv, inputs=[qt_ann_session, qa_all], outputs=[qa_dlf])
                qa_merge.click(fn=qn._update_main_df_from_annotations,
                               inputs=[qt_ann_session, qt_df, qt_tcol, qt_lcol], outputs=[qt_df, qa_mmsg])
                # Sync label choices when user edits the label list
                qa_labels.change(
                    fn=lambda s: gr.update(choices=[l.strip() for l in s.split(",") if l.strip()]),
                    inputs=[qa_labels], outputs=[qa_input],
                )

            # -- 4. Classification ----------------------------------------
            with gr.Tab("Step 4 · Classification"):
                qt_s4_md = gr.Markdown(t("qt.s4.title", "en"))
                with gr.Row():
                    qt_be = gr.Dropdown(choices=["openai", "anthropic", "ollama"], value="openai", label=t("common.backend", "en"))
                    qt_mod = gr.Textbox(label=t("common.model", "en"), value="gpt-4o-mini")
                    qt_key = gr.Textbox(label=t("common.api_key", "en"), type="password")
                with gr.Row():
                    qt_cls = gr.Textbox(label=t("qt.s4.classes", "en"), value="positive, negative, neutral", placeholder=t("qt.s4.classes_placeholder", "en"))

                with gr.Tabs():
                    with gr.Tab("Prompt Classification"):
                        qt_s4_design_md = gr.Markdown(t("qt.s4.design_title", "en"))
                        qt_task_desc = gr.Textbox(
                            label=t("qt.s4.task_desc", "en"), lines=3,
                            placeholder=t("qt.s4.task_desc_placeholder", "en"),
                        )
                        qt_defs = gr.Textbox(
                            label=t("qt.s4.class_defs", "en"), lines=3,
                            placeholder=t("qt.s4.class_defs_placeholder", "en"),
                        )
                        qt_exs = gr.Textbox(
                            label=t("qt.s4.positive_ex", "en"), lines=3,
                            placeholder=t("qt.s4.positive_ex_placeholder", "en"),
                        )
                        qt_excl = gr.Textbox(
                            label=t("qt.s4.negative_ex", "en"), lines=2,
                            placeholder=t("qt.s4.negative_ex_placeholder", "en"),
                        )
                        qt_gen_btn = gr.Button(t("qt.s4.generate_prompt", "en"), variant="primary")

                        qt_s4_prompt_md = gr.Markdown(t("qt.s4.current_prompt_title", "en"))
                        qt_prompt = gr.Textbox(
                            label=t("qt.s4.current_prompt", "en"), lines=10, interactive=True,
                            placeholder=t("qt.s4.current_prompt_placeholder", "en"),
                        )

                        qt_s4_opt_md = gr.Markdown(t("qt.s4.optimize_title", "en"))
                        with gr.Row():
                            qt_eval_btn = gr.Button(t("qt.s4.eval_btn", "en"), variant="secondary")
                            qt_copy_btn = gr.Button(t("qt.s4.copy_btn", "en"), variant="secondary", size="sm")
                        qt_eval_msg = gr.Textbox(label=t("qt.s4.eval_result", "en"), lines=8, interactive=False)
                        qt_s4_compare_md = gr.Markdown(t("qt.s4.compare_desc", "en"))
                        with gr.Row():
                            qt_v1 = gr.Textbox(label=t("qt.s4.variant_1", "en"), lines=6, interactive=True, scale=1)
                            qt_v2 = gr.Textbox(label=t("qt.s4.variant_2", "en"), lines=6, interactive=True, scale=1)
                            qt_v3 = gr.Textbox(label=t("qt.s4.variant_3", "en"), lines=6, interactive=True, scale=1)
                        with gr.Row():
                            qt_use1 = gr.Button(t("qt.s4.use_v1", "en"), variant="secondary", size="sm")
                            qt_use2 = gr.Button(t("qt.s4.use_v2", "en"), variant="secondary", size="sm")
                            qt_use3 = gr.Button(t("qt.s4.use_v3", "en"), variant="secondary", size="sm")
                        qt_test_btn = gr.Button(t("qt.s4.test_btn", "en"), variant="primary")
                        qt_test_sum = gr.Textbox(label=t("qt.s4.test_result", "en"), lines=6, interactive=False)
                        qt_test_detail = gr.Dataframe(
                            label=t("qt.s4.test_detail", "en"),
                            interactive=False, wrap=True,
                        )

                        qt_s4_run_md = gr.Markdown(t("qt.s4.run_title", "en"))
                        qt_cbtn = gr.Button(t("qt.s4.run_btn", "en"), variant="primary")
                        qt_csum = gr.Textbox(label=t("qt.s4.result", "en"), lines=4, interactive=False)
                        qt_cres = gr.Dataframe(label=t("qt.s4.detail", "en"), interactive=False)

                    with gr.Tab("Fine-tuning"):
                        qt_s4_ft_md = gr.Markdown(t("qt.s4.ft_explainer", "en"))
                        with gr.Row():
                            qt_ftm = gr.Dropdown(
                                choices=["roberta-base", "xlm-roberta-base", "bert-base-uncased", "bert-base-chinese"],
                                value="roberta-base", label=t("qt.s4.ft_model", "en"))
                            qt_ftbs = gr.Number(label="Batch Size", value=16, precision=0)
                            qt_ftep = gr.Number(label="Epochs", value=5, precision=0)
                            qt_ftlr = gr.Number(label="Learning Rate", value=2e-5)
                        qt_ftbtn = gr.Button(t("qt.s4.ft_start", "en"), variant="primary")
                        qt_ftsum = gr.Textbox(label=t("qt.s4.ft_result", "en"), lines=6, interactive=False)
                        qt_ftres = gr.Dataframe(label=t("qt.s4.ft_predictions", "en"), interactive=False)

                    with gr.Tab("API Fine-tuning"):
                        qt_s4_aft_md = gr.Markdown(t("qt.s4.aft_desc", "en"))
                        with gr.Row():
                            qt_aft_key = gr.Textbox(label=t("qt.s4.aft_key", "en"), type="password")
                            qt_aft_mod = gr.Dropdown(
                                choices=["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "gpt-3.5-turbo-0125"],
                                value="gpt-4o-mini-2024-07-18", label=t("qt.s4.aft_base_model", "en"))
                        with gr.Row():
                            qt_aft_ep = gr.Dropdown(
                                choices=["auto", "1", "2", "3", "4", "5"],
                                value="auto", label=t("qt.s4.aft_epochs", "en"))
                            qt_aft_sfx = gr.Textbox(label=t("qt.s4.aft_suffix", "en"), placeholder="my-classifier")
                        qt_aft_btn = gr.Button(t("qt.s4.aft_submit", "en"), variant="primary")
                        qt_aft_stat = gr.Textbox(label=t("qt.s4.aft_status", "en"), lines=6, interactive=False)
                        with gr.Row():
                            qt_aft_chk = gr.Button(t("qt.s4.aft_refresh", "en"), variant="secondary")
                            qt_aft_cnl = gr.Button(t("qt.s4.aft_cancel", "en"), variant="secondary")
                        qt_aft_res = gr.Dataframe(label=t("qt.s4.aft_predictions", "en"), interactive=False)
                        qt_aft_jid = gr.State("")

                # -- Prompt workflow events --
                qt_gen_btn.click(fn=qn._generate_smart_prompt,
                                 inputs=[qt_task_desc, qt_cls, qt_defs, qt_exs, qt_excl,
                                         qt_be, qt_mod, qt_key],
                                 outputs=[qt_prompt])
                qt_eval_btn.click(fn=qn._evaluate_prompt,
                                  inputs=[qt_prompt, qt_cls, qt_be, qt_mod, qt_key],
                                  outputs=[qt_eval_msg])
                qt_copy_btn.click(fn=lambda p: (p, p, p),
                                  inputs=[qt_prompt],
                                  outputs=[qt_v1, qt_v2, qt_v3])
                qt_test_btn.click(fn=qn._test_variants,
                                  inputs=[qt_df, qt_tcol, qt_lcol,
                                          qt_prompt, qt_v1, qt_v2, qt_v3,
                                          qt_cls, qt_be, qt_mod, qt_key],
                                  outputs=[qt_test_sum, qt_test_detail])
                def _adopt(v, name):
                    if not v or not v.strip():
                        gr.Warning(f"{name} 为空，无法采用。")
                        raise gr.Error(f"{name} 为空")
                    gr.Info(f"✅ 已采用{name}作为当前 Prompt")
                    return v

                qt_use1.click(fn=lambda v: _adopt(v, "版本 1"), inputs=[qt_v1], outputs=[qt_prompt])
                qt_use2.click(fn=lambda v: _adopt(v, "版本 2"), inputs=[qt_v2], outputs=[qt_prompt])
                qt_use3.click(fn=lambda v: _adopt(v, "版本 3"), inputs=[qt_v3], outputs=[qt_prompt])
                qt_cbtn.click(fn=qn._run_classification,
                              inputs=[qt_df, qt_tcol, qt_lcol, qt_cls, qt_prompt,
                                      qt_be, qt_mod, qt_key],
                              outputs=[qt_csum, qt_result_df])
                qt_result_df.change(fn=lambda x: x, inputs=[qt_result_df], outputs=[qt_cres])

                def _ft_wrap(df, tc, lc, m, bs, ep, lr):
                    s, r = qn._run_finetune(df, tc, lc, m, bs, ep, lr)
                    return s, r, r
                qt_ftbtn.click(fn=_ft_wrap,
                               inputs=[qt_df, qt_tcol, qt_lcol, qt_ftm, qt_ftbs, qt_ftep, qt_ftlr],
                               outputs=[qt_ftsum, qt_ftres, qt_result_df])

                # API Fine-tuning events
                qt_aft_btn.click(fn=qn._start_api_finetune,
                                 inputs=[qt_df, qt_tcol, qt_lcol, qt_cls, qt_defs,
                                         qt_aft_key, qt_aft_mod, qt_aft_ep, qt_aft_sfx],
                                 outputs=[qt_aft_stat, qt_aft_jid])
                qt_aft_chk.click(fn=qn._check_api_ft_status,
                                 inputs=[qt_aft_jid, qt_df, qt_tcol, qt_aft_key],
                                 outputs=[qt_aft_stat, qt_aft_res, qt_result_df])
                qt_aft_cnl.click(fn=qn._cancel_api_ft_job,
                                 inputs=[qt_aft_jid, qt_aft_key],
                                 outputs=[qt_aft_stat])

            # -- 5. Evaluation --------------------------------------------
            with gr.Tab("Step 5 · Evaluation"):
                qt_s5_md = gr.Markdown(t("qt.s5.title", "en"))
                qt_ebtn = gr.Button(t("qt.s5.run_btn", "en"), variant="primary")
                qt_eout = gr.Textbox(label=t("qt.s5.report", "en"), lines=18, interactive=False)
                qt_ebtn.click(fn=qn._evaluate_results, inputs=[qt_result_df, qt_df, qt_lcol],
                              outputs=[qt_eout])

            # -- 6. Export ------------------------------------------------
            with gr.Tab("Step 6 · Export"):
                qt_s6_md = gr.Markdown(t("qt.s6.title", "en"))
                qt_xbtn = gr.Button(t("qt.s6.export_btn", "en"), variant="primary")
                qt_xfile = gr.File(label=t("qt.s6.file", "en"))
                qt_xbtn.click(fn=qn._export_results, inputs=[qt_result_df], outputs=[qt_xfile])

                qt_log_btn = gr.Button(t("toolbox.export_log", "en"), variant="secondary")
                qt_log_file = gr.File(label="Pipeline Log")
                qt_log_btn.click(fn=qn._export_pipeline_log,
                                 inputs=[qt_result_df, qt_df, qt_lcol],
                                 outputs=[qt_log_file])

        # ==================================================================
        # QualiKit
        # ==================================================================
        with gr.Tab("QualiKit"):

            # ---- shared state ----
            ql_raw_text = gr.State("")
            ql_segments = gr.State([])
            ql_rqs = gr.State([])          # list[ResearchQuestion]
            ql_ext_session = gr.State(None)
            ql_deident_sess = gr.State(None)
            ql_deident_res = gr.State(None)

            # ==========================================================
            # Step 1 · Upload & Segment
            # ==========================================================
            with gr.Tab("Step 1 · Upload & Segment"):
                ql_s1_md = gr.Markdown(t("ql.s1.title", "en"))
                with gr.Row():
                    ql_file = gr.File(label=t("ql.s1.upload", "en"), file_types=[".txt"])
                    with gr.Column(scale=0, min_width=140):
                        ql_tpl_btn = gr.Button(t("ql.s1.download_example", "en"), variant="secondary", size="sm")
                        ql_tpl_file = gr.File(label=t("ql.s1.example_file", "en"), visible=False)
                ql_text_preview = gr.Textbox(
                    label=t("ql.s1.preview", "en"), lines=8, interactive=False,
                    placeholder=t("ql.s1.preview_placeholder", "en"),
                )
                ql_load_msg = gr.Textbox(label="", interactive=False, show_label=False)

                ql_s1_seg_md = gr.Markdown(t("ql.s1.segment_title", "en"))
                with gr.Row():
                    ql_seg_mode = gr.Radio(
                        choices=["paragraph", "sentence", "context_window"],
                        value="paragraph", label=t("ql.s1.mode", "en"),
                        info=t("ql.s1.mode_info", "en"),
                    )
                    ql_seg_cw = gr.Slider(
                        minimum=1, maximum=5, value=2, step=1,
                        label=t("ql.s1.context_window", "en"), info=t("ql.s1.context_info", "en"),
                    )
                ql_seg_btn = gr.Button(t("ql.s1.segment_btn", "en"), variant="primary")
                ql_seg_msg = gr.Textbox(label="", interactive=False, show_label=False)
                ql_seg_tbl = gr.Dataframe(label=t("ql.s1.segment_preview", "en"), interactive=False)

                # -- events --
                ql_tpl_btn.click(fn=ql._get_interview_template, outputs=[ql_tpl_file])
                ql_file.change(
                    fn=ql._load_raw_text, inputs=[ql_file],
                    outputs=[ql_raw_text, ql_text_preview, ql_load_msg],
                )
                ql_seg_btn.click(
                    fn=ql._segment_and_preview,
                    inputs=[ql_raw_text, ql_seg_mode, ql_seg_cw],
                    outputs=[ql_segments, ql_seg_tbl, ql_seg_msg],
                )

            # ==========================================================
            # Step 2 · De-identification
            # ==========================================================
            with gr.Tab("Step 2 · De-identification"):
                ql_s2_md = gr.Markdown(t("ql.s2.title", "en"))
                with gr.Row():
                    ql_dent = gr.Textbox(
                        label=t("ql.s2.entity_types", "en"),
                        value="PERSON, ORG, LOCATION, DATE, PHONE, EMAIL, URL, ID_CARD",
                    )
                    ql_dstrat = gr.Dropdown(
                        choices=["placeholder", "category", "redact"],
                        value="placeholder", label=t("ql.s2.strategy", "en"),
                    )
                ql_dbtn = gr.Button(t("ql.s2.run_btn", "en"), variant="primary")
                ql_dstats = gr.Textbox(label=t("ql.s2.stats", "en"), interactive=False)

                ql_d_progress = gr.Textbox(label=t("ql.s2.progress", "en"), interactive=False)
                ql_d_rev_tbl = gr.Dataframe(label=t("ql.s2.review_table", "en"), interactive=False)

                ql_s2_detail_md = gr.Markdown("---\n" + t("ql.s2.detail_title", "en"))
                with gr.Row():
                    ql_d_idx = gr.Number(label=t("ql.s2.index", "en"), precision=0, value=0, minimum=0)
                    ql_d_acc = gr.Button(t("ql.s2.accept", "en"), variant="primary", size="sm")
                    ql_d_rej = gr.Button(t("ql.s2.reject", "en"), variant="secondary", size="sm")
                ql_d_detail = gr.HTML(label=t("ql.s2.detail", "en"))
                with gr.Row():
                    ql_d_edit_text = gr.Textbox(label=t("ql.s2.custom_text", "en"), placeholder=t("ql.s2.custom_placeholder", "en"), scale=3)
                    ql_d_edit_btn = gr.Button(t("ql.s2.edit", "en"), variant="secondary", size="sm", scale=0)

                ql_s2_bulk_md = gr.Markdown(t("ql.s2.bulk_title", "en"))
                with gr.Row():
                    ql_daa = gr.Button(t("ql.s2.accept_all", "en"), variant="secondary")
                    ql_dah = gr.Button(t("ql.s2.accept_high", "en"), variant="secondary")
                    ql_dapp = gr.Button(t("ql.s2.apply", "en"), variant="primary")
                ql_dmsg = gr.Textbox(label="", interactive=False, show_label=False)

                # -- events --
                _d_rev_out = [ql_deident_sess, ql_d_rev_tbl, ql_d_progress]

                ql_dbtn.click(
                    fn=ql._deident_segments_v2,
                    inputs=[ql_segments, ql_dent, ql_dstrat],
                    outputs=[ql_deident_sess, ql_dstats, ql_d_rev_tbl,
                             ql_d_progress, ql_deident_res],
                )
                # Per-item review
                ql_d_idx.change(
                    fn=ql._deident_show_detail,
                    inputs=[ql_deident_sess, ql_d_idx, ql_segments],
                    outputs=[ql_d_detail],
                )
                ql_d_acc.click(
                    fn=ql._deident_accept_one,
                    inputs=[ql_deident_sess, ql_d_idx],
                    outputs=_d_rev_out,
                ).then(
                    fn=ql._deident_show_detail,
                    inputs=[ql_deident_sess, ql_d_idx, ql_segments],
                    outputs=[ql_d_detail],
                )
                ql_d_rej.click(
                    fn=ql._deident_reject_one,
                    inputs=[ql_deident_sess, ql_d_idx],
                    outputs=_d_rev_out,
                ).then(
                    fn=ql._deident_show_detail,
                    inputs=[ql_deident_sess, ql_d_idx, ql_segments],
                    outputs=[ql_d_detail],
                )
                ql_d_edit_btn.click(
                    fn=ql._deident_edit_one,
                    inputs=[ql_deident_sess, ql_d_idx, ql_d_edit_text],
                    outputs=_d_rev_out,
                ).then(
                    fn=ql._deident_show_detail,
                    inputs=[ql_deident_sess, ql_d_idx, ql_segments],
                    outputs=[ql_d_detail],
                )
                # Bulk operations
                ql_daa.click(
                    fn=ql._deident_accept_all_v2, inputs=[ql_deident_sess],
                    outputs=_d_rev_out,
                )
                ql_dah.click(
                    fn=ql._deident_accept_high_v2, inputs=[ql_deident_sess],
                    outputs=_d_rev_out,
                )
                ql_dapp.click(
                    fn=ql._deident_apply_to_segments,
                    inputs=[ql_deident_sess, ql_segments],
                    outputs=[ql_segments, ql_dmsg],
                )

            # ==========================================================
            # Step 3 · Research Framework
            # ==========================================================
            with gr.Tab("Step 3 · Research Framework"):
                ql_s3_md = gr.Markdown(t("ql.s3.title", "en"))

                ql_s3_rq_md = gr.Markdown(t("ql.s3.rq_title", "en"))
                ql_rq_tbl = gr.Dataframe(
                    headers=["RQ ID", "Description"],
                    datatype=["str", "str"],
                    row_count=(2, "dynamic"),
                    col_count=(2, "fixed"),
                    interactive=True,
                    label=t("ql.s3.rq_table", "en"),
                    value=[["RQ1", ""], ["RQ2", ""]],
                )

                ql_s3_sub_md = gr.Markdown(t("ql.s3.sub_title", "en"))
                ql_sub_tbl = gr.Dataframe(
                    headers=["Parent RQ", "Sub-theme Name"],
                    datatype=["str", "str"],
                    row_count=(4, "dynamic"),
                    col_count=(2, "fixed"),
                    interactive=True,
                    label=t("ql.s3.sub_table", "en"),
                    value=[["RQ1", ""], ["RQ1", ""], ["RQ2", ""], ["RQ2", ""]],
                )

                ql_rq_confirm_btn = gr.Button(t("ql.s3.confirm_btn", "en"), variant="primary")
                ql_rq_summary = gr.Textbox(label=t("ql.s3.parsed", "en"), lines=5, interactive=False)

                ql_s3_sug_md = gr.Markdown(t("ql.s3.suggest_title", "en"))
                with gr.Row():
                    ql_sug_be = gr.Dropdown(
                        choices=["openai", "anthropic", "ollama"],
                        value="openai", label=t("ql.s3.backend", "en"),
                    )
                    ql_sug_mod = gr.Textbox(label=t("ql.s3.model", "en"), value="gpt-4o-mini")
                    ql_sug_key = gr.Textbox(label=t("ql.s3.api_key", "en"), type="password")
                ql_sug_btn = gr.Button(t("ql.s3.suggest_btn", "en"), variant="secondary")
                ql_sug_out = gr.Textbox(label=t("ql.s3.suggest_result", "en"), lines=10, interactive=False)

                # -- events (dropdown updates wired after Step 5 components) --
                ql_sug_btn.click(
                    fn=ql._suggest_sub_themes,
                    inputs=[ql_segments, ql_rq_tbl, ql_sug_be, ql_sug_mod, ql_sug_key],
                    outputs=[ql_sug_out],
                )

            # ==========================================================
            # Step 4 · LLM Coding
            # ==========================================================
            with gr.Tab("Step 4 · LLM Coding"):
                ql_s4_md = gr.Markdown(t("ql.s4.title", "en"))
                with gr.Row():
                    ql_ext_be = gr.Dropdown(
                        choices=["openai", "anthropic", "ollama"],
                        value="openai", label=t("ql.s4.backend", "en"),
                    )
                    ql_ext_mod = gr.Textbox(label=t("ql.s4.model", "en"), value="gpt-4o-mini")
                    ql_ext_key = gr.Textbox(label=t("ql.s4.api_key", "en"), type="password")
                ql_ext_btn = gr.Button(t("ql.s4.run_btn", "en"), variant="primary")
                ql_ext_msg = gr.Textbox(label=t("ql.s4.result", "en"), lines=4, interactive=False)
                ql_ext_tbl = gr.Dataframe(label=t("ql.s4.detail", "en"), interactive=False)

                # events wired in cross-tab section below (needs Step 5 components)

            # ==========================================================
            # Step 5 · Review
            # ==========================================================
            with gr.Tab("Step 5 · Review"):
                ql_s5_md = gr.Markdown(t("ql.s5.title", "en"))
                ql_rev_stats = gr.Textbox(label=t("ql.s5.stats", "en"), interactive=False)
                ql_rev_tbl = gr.Dataframe(label=t("ql.s5.table", "en"), interactive=False)

                ql_s5_detail_md = gr.Markdown("---\n" + t("ql.s5.detail_title", "en"))
                with gr.Row():
                    ql_rev_idx = gr.Number(label=t("ql.s5.index", "en"), precision=0, value=0, minimum=0)
                    ql_rev_acc = gr.Button(t("ql.s5.accept", "en"), variant="primary", size="sm")
                    ql_rev_rej = gr.Button(t("ql.s5.reject", "en"), variant="secondary", size="sm")
                ql_rev_ctx = gr.HTML(label=t("ql.s5.detail", "en"))
                with gr.Row():
                    ql_rev_edit_rq = gr.Textbox(label=t("ql.s5.edit_rq", "en"), placeholder="RQ2", scale=1)
                    ql_rev_edit_sub = gr.Textbox(label=t("ql.s5.edit_sub", "en"), placeholder="income change", scale=2)
                    ql_rev_edit_btn = gr.Button(t("ql.s5.edit", "en"), variant="secondary", size="sm", scale=0)

                ql_s5_bulk_md = gr.Markdown(t("ql.s5.bulk_title", "en"))
                with gr.Row():
                    ql_rev_thr = gr.Slider(
                        minimum=0.5, maximum=1.0, value=0.85, step=0.05,
                        label=t("ql.s5.threshold", "en"),
                    )
                    ql_rev_bulk = gr.Button(t("ql.s5.bulk_accept", "en"), variant="secondary")

                ql_s5_manual_md = gr.Markdown(t("ql.s5.manual_title", "en"))
                ql_man_preview = gr.HTML(
                    value=t("ql.s5.manual_preview_default", "en"),
                )
                with gr.Row():
                    ql_man_seg = gr.Number(label=t("ql.s5.seg_id", "en"), precision=0, value=1, minimum=1)
                    ql_man_rq = gr.Dropdown(label=t("ql.s5.rq_label", "en"), choices=[], interactive=True)
                    ql_man_sub = gr.Dropdown(label=t("ql.s5.sub_theme", "en"), choices=[], interactive=True)
                    ql_man_btn = gr.Button(t("ql.s5.add", "en"), variant="secondary", size="sm")

            # ==========================================================
            # Step 6 · Export
            # ==========================================================
            with gr.Tab("Step 6 · Export"):
                ql_s6_md = gr.Markdown(t("ql.s6.title", "en"))
                ql_xl_btn = gr.Button(t("ql.s6.export_btn", "en"), variant="primary")
                ql_xl_file = gr.File(label=t("ql.s6.file", "en"))
                ql_xl_msg = gr.Textbox(label="", interactive=False, show_label=False)

                ql_xl_btn.click(
                    fn=ql._ext_export_excel, inputs=[ql_ext_session],
                    outputs=[ql_xl_file, ql_xl_msg],
                )

                ql_log_btn = gr.Button(t("toolbox.export_log", "en"), variant="secondary")
                ql_log_file = gr.File(label="Pipeline Log")
                ql_log_btn.click(
                    fn=ql._export_pipeline_log,
                    inputs=[ql_ext_session, ql_rqs, ql_ext_session],
                    outputs=[ql_log_file],
                )

            # ==========================================================
            # Cross-tab event wiring
            # (placed here so all components are defined)
            # ==========================================================

            # Step 3 → confirm framework → populate dropdowns in Step 5
            ql_rq_confirm_btn.click(
                fn=ql._confirm_rq_framework,
                inputs=[ql_rq_tbl, ql_sub_tbl],
                outputs=[ql_rqs, ql_rq_summary, ql_man_rq, ql_man_sub],
            )

            # Step 4 → extraction outputs to Step 5 review table
            ql_ext_btn.click(
                fn=ql._run_extraction_v2,
                inputs=[ql_raw_text, ql_segments, ql_rqs,
                        ql_ext_be, ql_ext_mod, ql_ext_key],
                outputs=[ql_ext_session, ql_ext_msg, ql_ext_tbl,
                         ql_rev_tbl, ql_rev_stats],
            )

            # Step 5 — review events
            _rev_out = [ql_ext_session, ql_rev_tbl, ql_rev_stats]

            ql_rev_idx.change(
                fn=ql._ext_show_context,
                inputs=[ql_ext_session, ql_rev_idx, ql_raw_text],
                outputs=[ql_rev_ctx],
            )
            ql_rev_acc.click(
                fn=ql._ext_accept, inputs=[ql_ext_session, ql_rev_idx],
                outputs=_rev_out,
            ).then(
                fn=ql._ext_show_context,
                inputs=[ql_ext_session, ql_rev_idx, ql_raw_text],
                outputs=[ql_rev_ctx],
            )
            ql_rev_rej.click(
                fn=ql._ext_reject, inputs=[ql_ext_session, ql_rev_idx],
                outputs=_rev_out,
            ).then(
                fn=ql._ext_show_context,
                inputs=[ql_ext_session, ql_rev_idx, ql_raw_text],
                outputs=[ql_rev_ctx],
            )
            ql_rev_edit_btn.click(
                fn=ql._ext_edit,
                inputs=[ql_ext_session, ql_rev_idx, ql_rev_edit_rq, ql_rev_edit_sub],
                outputs=_rev_out,
            ).then(
                fn=ql._ext_show_context,
                inputs=[ql_ext_session, ql_rev_idx, ql_raw_text],
                outputs=[ql_rev_ctx],
            )
            ql_rev_bulk.click(
                fn=ql._ext_accept_all_high,
                inputs=[ql_ext_session, ql_rev_thr],
                outputs=_rev_out,
            )

            # Manual add — segment preview on ID change
            ql_man_seg.change(
                fn=ql._preview_segment_by_id,
                inputs=[ql_segments, ql_man_seg],
                outputs=[ql_man_preview],
            )
            # Manual add — cascading sub-theme dropdown on RQ change
            ql_man_rq.change(
                fn=ql._get_sub_themes_for_rq,
                inputs=[ql_rqs, ql_man_rq],
                outputs=[ql_man_sub],
            )
            ql_man_btn.click(
                fn=ql._ext_add_manual,
                inputs=[ql_ext_session, ql_man_seg, ql_man_rq, ql_man_sub],
                outputs=_rev_out,
            )

        # ==================================================================
        # Toolbox
        # ==================================================================
        with gr.Tab(t("toolbox.title", "en")) as toolbox_tab:

            tb_intro_md = gr.Markdown(t("toolbox.description", "en"))

            # ---------- ICR Calculator ----------
            with gr.Tab(t("toolbox.icr_tab", "en")) as tb_icr_tab:
                tb_icr_md = gr.Markdown(t("icr.description", "en"))
                tb_icr_file = gr.File(label=t("toolbox.icr_upload", "en"), file_types=[".csv"])
                tb_icr_info = gr.Textbox(label=t("toolbox.icr_file_info", "en"), interactive=False, lines=1)
                tb_icr_cols = gr.CheckboxGroup(
                    choices=[], label=t("toolbox.icr_select_cols", "en"),
                    info=t("toolbox.icr_select_cols_info", "en"),
                )
                tb_icr_mode = gr.Radio(
                    choices=["single-label", "multi-label"],
                    value="single-label",
                    label=t("toolbox.icr_mode", "en"),
                    info=t("toolbox.icr_mode_info", "en"),
                )
                tb_icr_btn = gr.Button(t("icr.compute_btn", "en"), variant="primary")
                tb_icr_out = gr.Textbox(label=t("icr.report", "en"), lines=18, interactive=False)

                tb_icr_file.change(
                    fn=tb._icr_on_upload,
                    inputs=[tb_icr_file],
                    outputs=[tb_icr_cols, tb_icr_info],
                )
                tb_icr_btn.click(
                    fn=tb._compute_icr,
                    inputs=[tb_icr_file, tb_icr_cols, tb_icr_mode],
                    outputs=[tb_icr_out],
                )

            # ---------- Consensus Coding ----------
            with gr.Tab(t("toolbox.consensus_tab", "en")) as tb_con_tab:
                tb_con_md = gr.Markdown(t("consensus.description", "en"))
                with gr.Row():
                    tb_con_file = gr.File(label=t("toolbox.data_file", "en"), file_types=[".csv"])
                    tb_con_tcol = gr.Textbox(label=t("toolbox.text_col", "en"), value="text")
                tb_con_themes = gr.Textbox(
                    label=t("toolbox.themes_input", "en"),
                    placeholder="theme1: description\ntheme2: description\n...",
                    lines=4,
                )

                # Dynamic LLM slots (up to 5, first 2 visible)
                tb_con_llm_rows = []   # list of (gr.Row, backend, model, key)
                tb_con_backends = []
                tb_con_models = []
                tb_con_keys = []
                _defaults = [
                    ("openai", "gpt-4o-mini"),
                    ("anthropic", "claude-sonnet-4-20250514"),
                    ("openai", "gpt-4o"),
                    ("ollama", ""),
                    ("anthropic", ""),
                ]
                for idx in range(tb.MAX_LLM_SLOTS):
                    visible = idx < 2
                    be_default, mod_default = _defaults[idx]
                    with gr.Row(visible=visible) as llm_row:
                        be = gr.Dropdown(
                            choices=["openai", "anthropic", "ollama"],
                            value=be_default,
                            label=f"Backend {idx + 1}",
                        )
                        md = gr.Textbox(label=f"Model {idx + 1}", value=mod_default if visible else "")
                        ky = gr.Textbox(label=f"API Key {idx + 1}", type="password")
                    tb_con_llm_rows.append(llm_row)
                    tb_con_backends.append(be)
                    tb_con_models.append(md)
                    tb_con_keys.append(ky)

                tb_con_n_llms = gr.State(2)  # tracks how many slots are visible

                with gr.Row():
                    tb_con_add = gr.Button(t("toolbox.add_llm", "en"), variant="secondary", size="sm")
                    tb_con_rm = gr.Button(t("toolbox.remove_llm", "en"), variant="secondary", size="sm")

                def _add_llm_slot(n):
                    n = min(n + 1, tb.MAX_LLM_SLOTS)
                    updates = [gr.update(visible=(i < n)) for i in range(tb.MAX_LLM_SLOTS)]
                    return [n] + updates

                def _remove_llm_slot(n):
                    n = max(n - 1, 2)
                    updates = [gr.update(visible=(i < n)) for i in range(tb.MAX_LLM_SLOTS)]
                    return [n] + updates

                tb_con_add.click(
                    fn=_add_llm_slot,
                    inputs=[tb_con_n_llms],
                    outputs=[tb_con_n_llms] + tb_con_llm_rows,
                )
                tb_con_rm.click(
                    fn=_remove_llm_slot,
                    inputs=[tb_con_n_llms],
                    outputs=[tb_con_n_llms] + tb_con_llm_rows,
                )

                tb_con_btn = gr.Button(t("consensus.run_btn", "en"), variant="primary")
                tb_con_summary = gr.Textbox(label=t("consensus.summary", "en"), lines=10, interactive=False)
                tb_con_results = gr.Dataframe(label=t("consensus.results", "en"), interactive=False)
                tb_con_agreement = gr.Textbox(label=t("consensus.agreement", "en"), lines=4, interactive=False)

                # Collect all backend/model/key inputs in order
                _con_inputs = [tb_con_file, tb_con_tcol, tb_con_themes]
                for i in range(tb.MAX_LLM_SLOTS):
                    _con_inputs.extend([tb_con_backends[i], tb_con_models[i], tb_con_keys[i]])

                tb_con_btn.click(
                    fn=tb._run_standalone_consensus,
                    inputs=_con_inputs,
                    outputs=[tb_con_summary, tb_con_results, tb_con_agreement],
                )

            # ---------- Methods Generator ----------
            with gr.Tab(t("toolbox.methods_tab", "en")) as tb_meth_tab:
                tb_meth_md = gr.Markdown(t("methods.description", "en"))

                # Primary: import log
                gr.Markdown(f"### {t('toolbox.import_log', 'en')}")
                tb_meth_log = gr.File(
                    label=t("toolbox.import_log", "en"),
                    file_types=[".json"],
                )
                tb_meth_log_btn = gr.Button(t("methods.generate_btn", "en"), variant="primary")

                tb_meth_en = gr.Textbox(label=t("methods.text_en", "en"), lines=8, interactive=True)
                tb_meth_zh = gr.Textbox(label=t("methods.text_zh", "en"), lines=8, interactive=True)
                gr.Markdown(t("methods.copy_hint", "en"))

                tb_meth_log_btn.click(
                    fn=tb._generate_methods_from_log,
                    inputs=[tb_meth_log],
                    outputs=[tb_meth_en, tb_meth_zh],
                )

                # Fallback: manual form
                with gr.Accordion(t("toolbox.manual_input", "en"), open=False):
                    tb_meth_type = gr.Radio(
                        choices=["QuantiKit", "QualiKit"],
                        value="QuantiKit",
                        label=t("toolbox.pipeline_type", "en"),
                    )
                    # QuantiKit fields
                    with gr.Group(visible=True) as tb_qt_group:
                        with gr.Row():
                            tb_qt_ns = gr.Number(label="N samples", value=0, precision=0)
                            tb_qt_nc = gr.Number(label="N classes", value=0, precision=0)
                        tb_qt_labels = gr.Textbox(label="Class labels (comma-separated)", value="")
                        tb_qt_model = gr.Textbox(label="Model name", value="")
                        with gr.Row():
                            tb_qt_acc = gr.Number(label="Accuracy", value=0, precision=3)
                            tb_qt_f1 = gr.Number(label="Macro F1", value=0, precision=3)
                            tb_qt_kappa = gr.Number(label="Cohen's Kappa", value=0, precision=3)

                    # QualiKit fields
                    with gr.Group(visible=False) as tb_ql_group:
                        with gr.Row():
                            tb_ql_nseg = gr.Number(label="N segments", value=0, precision=0)
                            tb_ql_nth = gr.Number(label="N themes", value=0, precision=0)
                        tb_ql_themes = gr.Textbox(label="Theme names (comma-separated)", value="")
                        tb_ql_model = gr.Textbox(label="Model name", value="")
                        tb_ql_consensus = gr.Checkbox(label="Consensus coding used", value=False)
                        tb_ql_ncon = gr.Number(label="N consensus models", value=0, precision=0)

                    def _toggle_pipeline_fields(choice):
                        return (
                            gr.update(visible=choice == "QuantiKit"),
                            gr.update(visible=choice == "QualiKit"),
                        )

                    tb_meth_type.change(
                        fn=_toggle_pipeline_fields,
                        inputs=[tb_meth_type],
                        outputs=[tb_qt_group, tb_ql_group],
                    )

                    tb_meth_form_btn = gr.Button(t("methods.generate_btn", "en"), variant="secondary")
                    tb_meth_form_btn.click(
                        fn=tb._generate_methods_from_form,
                        inputs=[tb_meth_type,
                                tb_qt_ns, tb_qt_nc, tb_qt_labels, tb_qt_model,
                                tb_qt_acc, tb_qt_f1, tb_qt_kappa,
                                tb_ql_nseg, tb_ql_nth, tb_ql_themes, tb_ql_model,
                                tb_ql_consensus, tb_ql_ncon],
                        outputs=[tb_meth_en, tb_meth_zh],
                    )

        # ==================================================================
        # Language switching
        # ==================================================================

        def _switch_language(choice):
            """Switch all UI text to the selected language."""
            lang = "zh" if choice == "中文" else "en"
            return [
                # ql_lang state
                lang,
                # Landing page
                _build_landing(lang),
                # -- QualiKit Step 1 --
                t("ql.s1.title", lang),                                  # ql_s1_md
                gr.update(label=t("ql.s1.upload", lang)),                # ql_file
                gr.update(value=t("ql.s1.download_example", lang)),      # ql_tpl_btn
                gr.update(label=t("ql.s1.example_file", lang)),          # ql_tpl_file
                gr.update(
                    label=t("ql.s1.preview", lang),
                    placeholder=t("ql.s1.preview_placeholder", lang),
                ),                                                       # ql_text_preview
                t("ql.s1.segment_title", lang),                          # ql_s1_seg_md
                gr.update(
                    label=t("ql.s1.mode", lang),
                    info=t("ql.s1.mode_info", lang),
                ),                                                       # ql_seg_mode
                gr.update(
                    label=t("ql.s1.context_window", lang),
                    info=t("ql.s1.context_info", lang),
                ),                                                       # ql_seg_cw
                gr.update(value=t("ql.s1.segment_btn", lang)),           # ql_seg_btn
                gr.update(label=t("ql.s1.segment_preview", lang)),       # ql_seg_tbl
                # -- QualiKit Step 2 --
                t("ql.s2.title", lang),                                  # ql_s2_md
                gr.update(label=t("ql.s2.entity_types", lang)),          # ql_dent
                gr.update(label=t("ql.s2.strategy", lang)),              # ql_dstrat
                gr.update(value=t("ql.s2.run_btn", lang)),               # ql_dbtn
                gr.update(label=t("ql.s2.stats", lang)),                 # ql_dstats
                gr.update(label=t("ql.s2.progress", lang)),              # ql_d_progress
                gr.update(label=t("ql.s2.review_table", lang)),          # ql_d_rev_tbl
                "---\n" + t("ql.s2.detail_title", lang),                 # ql_s2_detail_md
                gr.update(label=t("ql.s2.index", lang)),                 # ql_d_idx
                gr.update(value=t("ql.s2.accept", lang)),                # ql_d_acc
                gr.update(value=t("ql.s2.reject", lang)),                # ql_d_rej
                gr.update(label=t("ql.s2.detail", lang)),                # ql_d_detail
                gr.update(
                    label=t("ql.s2.custom_text", lang),
                    placeholder=t("ql.s2.custom_placeholder", lang),
                ),                                                       # ql_d_edit_text
                gr.update(value=t("ql.s2.edit", lang)),                  # ql_d_edit_btn
                t("ql.s2.bulk_title", lang),                             # ql_s2_bulk_md
                gr.update(value=t("ql.s2.accept_all", lang)),            # ql_daa
                gr.update(value=t("ql.s2.accept_high", lang)),           # ql_dah
                gr.update(value=t("ql.s2.apply", lang)),                 # ql_dapp
                # -- QualiKit Step 3 --
                t("ql.s3.title", lang),                                  # ql_s3_md
                t("ql.s3.rq_title", lang),                               # ql_s3_rq_md
                gr.update(label=t("ql.s3.rq_table", lang)),              # ql_rq_tbl
                t("ql.s3.sub_title", lang),                               # ql_s3_sub_md
                gr.update(label=t("ql.s3.sub_table", lang)),              # ql_sub_tbl
                gr.update(value=t("ql.s3.confirm_btn", lang)),            # ql_rq_confirm_btn
                gr.update(label=t("ql.s3.parsed", lang)),                 # ql_rq_summary
                t("ql.s3.suggest_title", lang),                           # ql_s3_sug_md
                gr.update(label=t("ql.s3.backend", lang)),                # ql_sug_be
                gr.update(label=t("ql.s3.model", lang)),                  # ql_sug_mod
                gr.update(label=t("ql.s3.api_key", lang)),                # ql_sug_key
                gr.update(value=t("ql.s3.suggest_btn", lang)),            # ql_sug_btn
                gr.update(label=t("ql.s3.suggest_result", lang)),         # ql_sug_out
                # -- QualiKit Step 4 --
                t("ql.s4.title", lang),                                  # ql_s4_md
                gr.update(label=t("ql.s4.backend", lang)),               # ql_ext_be
                gr.update(label=t("ql.s4.model", lang)),                 # ql_ext_mod
                gr.update(label=t("ql.s4.api_key", lang)),               # ql_ext_key
                gr.update(value=t("ql.s4.run_btn", lang)),               # ql_ext_btn
                gr.update(label=t("ql.s4.result", lang)),                # ql_ext_msg
                gr.update(label=t("ql.s4.detail", lang)),                # ql_ext_tbl
                # -- QualiKit Step 5 --
                t("ql.s5.title", lang),                                  # ql_s5_md
                gr.update(label=t("ql.s5.stats", lang)),                 # ql_rev_stats
                gr.update(label=t("ql.s5.table", lang)),                 # ql_rev_tbl
                "---\n" + t("ql.s5.detail_title", lang),                 # ql_s5_detail_md
                gr.update(label=t("ql.s5.index", lang)),                 # ql_rev_idx
                gr.update(value=t("ql.s5.accept", lang)),                # ql_rev_acc
                gr.update(value=t("ql.s5.reject", lang)),                # ql_rev_rej
                gr.update(label=t("ql.s5.detail", lang)),                # ql_rev_ctx
                gr.update(label=t("ql.s5.edit_rq", lang)),               # ql_rev_edit_rq
                gr.update(label=t("ql.s5.edit_sub", lang)),              # ql_rev_edit_sub
                gr.update(value=t("ql.s5.edit", lang)),                  # ql_rev_edit_btn
                t("ql.s5.bulk_title", lang),                             # ql_s5_bulk_md
                gr.update(label=t("ql.s5.threshold", lang)),             # ql_rev_thr
                gr.update(value=t("ql.s5.bulk_accept", lang)),           # ql_rev_bulk
                t("ql.s5.manual_title", lang),                           # ql_s5_manual_md
                t("ql.s5.manual_preview_default", lang),                 # ql_man_preview
                gr.update(label=t("ql.s5.seg_id", lang)),                # ql_man_seg
                gr.update(label=t("ql.s5.rq_label", lang)),              # ql_man_rq
                gr.update(label=t("ql.s5.sub_theme", lang)),             # ql_man_sub
                gr.update(value=t("ql.s5.add", lang)),                   # ql_man_btn
                # -- QualiKit Step 6 --
                t("ql.s6.title", lang),                                  # ql_s6_md
                gr.update(value=t("ql.s6.export_btn", lang)),            # ql_xl_btn
                gr.update(label=t("ql.s6.file", lang)),                  # ql_xl_file
                # -- QuantiKit Step 1 --
                t("qt.s1.title", lang),                                  # qt_s1_md
                gr.update(label=t("qt.s1.file", lang)),                  # qt_file
                gr.update(value=t("qt.s1.download_template", lang)),     # qt_tpl_btn
                gr.update(label=t("qt.s1.template", lang)),              # qt_tpl_file
                t("qt.s1.col_mapping_title", lang),                      # qt_s1_col_md
                gr.update(label=t("qt.s1.text_col", lang)),              # qt_tcol
                gr.update(label=t("qt.s1.label_col", lang)),             # qt_lcol
                gr.update(value=t("qt.s1.confirm_col", lang)),           # qt_col_btn
                gr.update(label=t("qt.s1.report", lang)),                # qt_summary
                gr.update(label=t("qt.s1.issues", lang)),                # qt_issues
                gr.update(label=t("qt.s1.preview", lang)),               # qt_preview
                gr.update(value=t("qt.s1.fix_btn", lang)),               # qt_fix_btn
                # -- QuantiKit Step 2 --
                t("qt.s2.title", lang),                                  # qt_s2_md
                gr.update(label=t("qt.s2.task_type", lang)),             # qt_task
                gr.update(label=t("qt.s2.num_classes", lang)),           # qt_ncls
                gr.update(label=t("qt.s2.target_f1", lang)),             # qt_f1
                gr.update(label=t("qt.s2.budget", lang)),                # qt_budget
                gr.update(value=t("qt.s2.recommend_btn", lang)),         # qt_rec_btn
                gr.update(label=t("qt.s2.features", lang)),              # qt_feat
                gr.update(label=t("qt.s2.annotation_budget", lang)),     # qt_bgt
                gr.update(label=t("qt.s2.recommendation", lang)),        # qt_rec
                gr.update(label=t("qt.s2.curve_plot", lang)),            # qt_curve_plot
                gr.update(label=t("qt.s2.key_points", lang)),            # qt_curve_tbl
                # -- QuantiKit Step 3 --
                t("qt.s3.title", lang),                                  # qt_s3_md
                gr.update(
                    label=t("qt.s3.labels", lang),
                    placeholder=t("qt.s3.labels_placeholder", lang),
                ),                                                       # qa_labels
                gr.update(label=t("qt.s3.shuffle", lang)),               # qa_shuf
                gr.update(value=t("qt.s3.create_session", lang)),        # qa_create
                gr.update(label=t("qt.s3.progress", lang)),              # qa_stats
                gr.update(label=t("qt.s3.current_pos", lang)),           # qa_idx
                gr.update(label=t("qt.s3.text_to_annotate", lang)),      # qa_text
                gr.update(label=t("qt.s3.label", lang)),                 # qa_input
                gr.update(value=t("qt.s3.annotate_btn", lang)),          # qa_sub
                gr.update(value=t("qt.s3.skip_btn", lang)),              # qa_skip
                gr.update(
                    label=t("qt.s3.note", lang),
                    placeholder=t("qt.s3.note_placeholder", lang),
                ),                                                       # qa_fnote
                gr.update(value=t("qt.s3.flag_btn", lang)),              # qa_flag
                gr.update(value=t("qt.s3.undo_btn", lang)),              # qa_undo
                gr.update(label=t("qt.s3.include_skipped", lang)),       # qa_all
                gr.update(value=t("qt.s3.preview_btn", lang)),           # qa_exp
                gr.update(value=t("qt.s3.download_csv", lang)),          # qa_dl
                gr.update(value=t("qt.s3.merge_btn", lang)),             # qa_merge
                gr.update(label=t("qt.s3.results", lang)),               # qa_tbl
                gr.update(label=t("qt.s3.download_file", lang)),         # qa_dlf
                # -- QuantiKit Step 4 --
                t("qt.s4.title", lang),                                  # qt_s4_md
                gr.update(label=t("common.backend", lang)),              # qt_be
                gr.update(label=t("common.model", lang)),                # qt_mod
                gr.update(label=t("common.api_key", lang)),              # qt_key
                gr.update(
                    label=t("qt.s4.classes", lang),
                    placeholder=t("qt.s4.classes_placeholder", lang),
                ),                                                       # qt_cls
                # -- Prompt sub-tab --
                t("qt.s4.design_title", lang),                           # qt_s4_design_md
                gr.update(
                    label=t("qt.s4.task_desc", lang),
                    placeholder=t("qt.s4.task_desc_placeholder", lang),
                ),                                                       # qt_task_desc
                gr.update(
                    label=t("qt.s4.class_defs", lang),
                    placeholder=t("qt.s4.class_defs_placeholder", lang),
                ),                                                       # qt_defs
                gr.update(
                    label=t("qt.s4.positive_ex", lang),
                    placeholder=t("qt.s4.positive_ex_placeholder", lang),
                ),                                                       # qt_exs
                gr.update(
                    label=t("qt.s4.negative_ex", lang),
                    placeholder=t("qt.s4.negative_ex_placeholder", lang),
                ),                                                       # qt_excl
                gr.update(value=t("qt.s4.generate_prompt", lang)),       # qt_gen_btn
                t("qt.s4.current_prompt_title", lang),                   # qt_s4_prompt_md
                gr.update(
                    label=t("qt.s4.current_prompt", lang),
                    placeholder=t("qt.s4.current_prompt_placeholder", lang),
                ),                                                       # qt_prompt
                t("qt.s4.optimize_title", lang),                         # qt_s4_opt_md
                gr.update(value=t("qt.s4.eval_btn", lang)),              # qt_eval_btn
                gr.update(value=t("qt.s4.copy_btn", lang)),              # qt_copy_btn
                gr.update(label=t("qt.s4.eval_result", lang)),           # qt_eval_msg
                t("qt.s4.compare_desc", lang),                           # qt_s4_compare_md
                gr.update(label=t("qt.s4.variant_1", lang)),             # qt_v1
                gr.update(label=t("qt.s4.variant_2", lang)),             # qt_v2
                gr.update(label=t("qt.s4.variant_3", lang)),             # qt_v3
                gr.update(value=t("qt.s4.use_v1", lang)),                # qt_use1
                gr.update(value=t("qt.s4.use_v2", lang)),                # qt_use2
                gr.update(value=t("qt.s4.use_v3", lang)),                # qt_use3
                gr.update(value=t("qt.s4.test_btn", lang)),              # qt_test_btn
                gr.update(label=t("qt.s4.test_result", lang)),           # qt_test_sum
                gr.update(label=t("qt.s4.test_detail", lang)),           # qt_test_detail
                t("qt.s4.run_title", lang),                              # qt_s4_run_md
                gr.update(value=t("qt.s4.run_btn", lang)),               # qt_cbtn
                gr.update(label=t("qt.s4.result", lang)),                # qt_csum
                gr.update(label=t("qt.s4.detail", lang)),                # qt_cres
                # -- Fine-tuning sub-tab --
                t("qt.s4.ft_explainer", lang),                           # qt_s4_ft_md
                gr.update(label=t("qt.s4.ft_model", lang)),              # qt_ftm
                gr.update(value=t("qt.s4.ft_start", lang)),              # qt_ftbtn
                gr.update(label=t("qt.s4.ft_result", lang)),             # qt_ftsum
                gr.update(label=t("qt.s4.ft_predictions", lang)),        # qt_ftres
                # -- API Fine-tuning sub-tab --
                t("qt.s4.aft_desc", lang),                               # qt_s4_aft_md
                gr.update(label=t("qt.s4.aft_key", lang)),               # qt_aft_key
                gr.update(label=t("qt.s4.aft_base_model", lang)),        # qt_aft_mod
                gr.update(label=t("qt.s4.aft_epochs", lang)),            # qt_aft_ep
                gr.update(label=t("qt.s4.aft_suffix", lang)),            # qt_aft_sfx
                gr.update(value=t("qt.s4.aft_submit", lang)),            # qt_aft_btn
                gr.update(label=t("qt.s4.aft_status", lang)),            # qt_aft_stat
                gr.update(value=t("qt.s4.aft_refresh", lang)),           # qt_aft_chk
                gr.update(value=t("qt.s4.aft_cancel", lang)),            # qt_aft_cnl
                gr.update(label=t("qt.s4.aft_predictions", lang)),       # qt_aft_res
                # -- QuantiKit Step 5 --
                t("qt.s5.title", lang),                                  # qt_s5_md
                gr.update(value=t("qt.s5.run_btn", lang)),               # qt_ebtn
                gr.update(label=t("qt.s5.report", lang)),                # qt_eout
                # -- QuantiKit Step 6 --
                t("qt.s6.title", lang),                                  # qt_s6_md
                gr.update(value=t("qt.s6.export_btn", lang)),            # qt_xbtn
                gr.update(label=t("qt.s6.file", lang)),                  # qt_xfile
                # -- Toolbox --
                t("toolbox.description", lang),                          # tb_intro_md
                t("icr.description", lang),                              # tb_icr_md
                gr.update(label=t("toolbox.icr_upload", lang)),          # tb_icr_file
                gr.update(label=t("toolbox.icr_select_cols", lang)),     # tb_icr_cols
                gr.update(label=t("toolbox.icr_mode", lang)),            # tb_icr_mode
                gr.update(value=t("icr.compute_btn", lang)),             # tb_icr_btn
                gr.update(label=t("icr.report", lang)),                  # tb_icr_out
                t("consensus.description", lang),                        # tb_con_md
                gr.update(label=t("toolbox.data_file", lang)),           # tb_con_file
                gr.update(label=t("toolbox.text_col", lang)),            # tb_con_tcol
                gr.update(label=t("toolbox.themes_input", lang)),        # tb_con_themes
                gr.update(value=t("toolbox.add_llm", lang)),             # tb_con_add
                gr.update(value=t("toolbox.remove_llm", lang)),          # tb_con_rm
                gr.update(value=t("consensus.run_btn", lang)),           # tb_con_btn
                gr.update(label=t("consensus.summary", lang)),           # tb_con_summary
                gr.update(label=t("consensus.results", lang)),           # tb_con_results
                gr.update(label=t("consensus.agreement", lang)),         # tb_con_agreement
                t("methods.description", lang),                          # tb_meth_md
                gr.update(label=t("toolbox.import_log", lang)),          # tb_meth_log
                gr.update(value=t("methods.generate_btn", lang)),        # tb_meth_log_btn
                gr.update(label=t("methods.text_en", lang)),             # tb_meth_en
                gr.update(label=t("methods.text_zh", lang)),             # tb_meth_zh
            ]

        _lang_outputs = [
            # ql_lang state
            ql_lang,
            # Landing
            landing_md,
            # Step 1
            ql_s1_md, ql_file, ql_tpl_btn, ql_tpl_file,
            ql_text_preview, ql_s1_seg_md, ql_seg_mode, ql_seg_cw,
            ql_seg_btn, ql_seg_tbl,
            # Step 2
            ql_s2_md, ql_dent, ql_dstrat, ql_dbtn, ql_dstats,
            ql_d_progress, ql_d_rev_tbl, ql_s2_detail_md,
            ql_d_idx, ql_d_acc, ql_d_rej, ql_d_detail,
            ql_d_edit_text, ql_d_edit_btn, ql_s2_bulk_md,
            ql_daa, ql_dah, ql_dapp,
            # Step 3
            ql_s3_md, ql_s3_rq_md, ql_rq_tbl, ql_s3_sub_md, ql_sub_tbl,
            ql_rq_confirm_btn, ql_rq_summary, ql_s3_sug_md,
            ql_sug_be, ql_sug_mod, ql_sug_key, ql_sug_btn, ql_sug_out,
            # Step 4
            ql_s4_md, ql_ext_be, ql_ext_mod, ql_ext_key,
            ql_ext_btn, ql_ext_msg, ql_ext_tbl,
            # Step 5
            ql_s5_md, ql_rev_stats, ql_rev_tbl, ql_s5_detail_md,
            ql_rev_idx, ql_rev_acc, ql_rev_rej, ql_rev_ctx,
            ql_rev_edit_rq, ql_rev_edit_sub, ql_rev_edit_btn,
            ql_s5_bulk_md, ql_rev_thr, ql_rev_bulk,
            ql_s5_manual_md, ql_man_preview, ql_man_seg,
            ql_man_rq, ql_man_sub, ql_man_btn,
            # Step 6
            ql_s6_md, ql_xl_btn, ql_xl_file,
            # ---- QuantiKit Step 1 ----
            qt_s1_md, qt_file, qt_tpl_btn, qt_tpl_file,
            qt_s1_col_md, qt_tcol, qt_lcol, qt_col_btn,
            qt_summary, qt_issues, qt_preview, qt_fix_btn,
            # ---- QuantiKit Step 2 ----
            qt_s2_md, qt_task, qt_ncls, qt_f1, qt_budget,
            qt_rec_btn, qt_feat, qt_bgt, qt_rec,
            qt_curve_plot, qt_curve_tbl,
            # ---- QuantiKit Step 3 ----
            qt_s3_md, qa_labels, qa_shuf, qa_create,
            qa_stats, qa_idx, qa_text, qa_input,
            qa_sub, qa_skip, qa_fnote, qa_flag, qa_undo,
            qa_all, qa_exp, qa_dl, qa_merge, qa_tbl, qa_dlf,
            # ---- QuantiKit Step 4 ----
            qt_s4_md, qt_be, qt_mod, qt_key, qt_cls,
            # Prompt sub-tab
            qt_s4_design_md, qt_task_desc, qt_defs, qt_exs, qt_excl,
            qt_gen_btn, qt_s4_prompt_md, qt_prompt, qt_s4_opt_md,
            qt_eval_btn, qt_copy_btn, qt_eval_msg, qt_s4_compare_md,
            qt_v1, qt_v2, qt_v3, qt_use1, qt_use2, qt_use3,
            qt_test_btn, qt_test_sum, qt_test_detail,
            qt_s4_run_md, qt_cbtn, qt_csum, qt_cres,
            # Fine-tuning sub-tab
            qt_s4_ft_md, qt_ftm, qt_ftbtn, qt_ftsum, qt_ftres,
            # API Fine-tuning sub-tab
            qt_s4_aft_md, qt_aft_key, qt_aft_mod, qt_aft_ep, qt_aft_sfx,
            qt_aft_btn, qt_aft_stat, qt_aft_chk, qt_aft_cnl, qt_aft_res,
            # ---- QuantiKit Step 5 ----
            qt_s5_md, qt_ebtn, qt_eout,
            # ---- QuantiKit Step 6 ----
            qt_s6_md, qt_xbtn, qt_xfile,
            # ---- Toolbox ----
            tb_intro_md,
            tb_icr_md, tb_icr_file, tb_icr_cols,
            tb_icr_mode, tb_icr_btn, tb_icr_out,
            tb_con_md, tb_con_file, tb_con_tcol, tb_con_themes,
            tb_con_add, tb_con_rm,
            tb_con_btn, tb_con_summary, tb_con_results, tb_con_agreement,
            tb_meth_md, tb_meth_log, tb_meth_log_btn, tb_meth_en, tb_meth_zh,
        ]

        lang_selector.change(
            fn=_switch_language,
            inputs=[lang_selector],
            outputs=_lang_outputs,
        )

    return app


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch(port: int = 7860, share: bool = False) -> None:
    """Launch the unified SocialSciKit app."""
    app = create_app()
    app.launch(server_port=port, share=share)
