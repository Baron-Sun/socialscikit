"""Gradio Web UI for QualiKit — qualitative coding workflow.

Steps:
1. Upload data -> validation + diagnostics
2. De-identification -> interactive review
3. Theme definition -> interactive review + overlap check
4. LLM coding -> confidence ranking -> interactive review
5. Export (excerpts table + co-occurrence matrix + memo)

Launch via CLI: ``socialscikit qualikit``
"""

from __future__ import annotations

import html as _html_escape
import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from socialscikit.core.data_diagnostics import format_report, generate_diagnostics
from socialscikit.core.data_loader import DataLoadError, get_template_path, load_file
from socialscikit.ui.i18n import t
from socialscikit.core.data_validator import DataValidator, apply_auto_fixes
from socialscikit.core.llm_client import LLMClient
from socialscikit.qualikit.coder import Coder, CodingResult
from socialscikit.qualikit.coding_reviewer import CodingReviewer, CodingReviewAction
from socialscikit.qualikit.confidence_ranker import ConfidenceRanker
from socialscikit.qualikit.deidentifier import Deidentifier
from socialscikit.qualikit.deident_reviewer import DeidentReviewer
from socialscikit.qualikit.exporter import Exporter
from socialscikit.qualikit.theme_definer import Theme, ThemeDefiner
from socialscikit.qualikit.extraction_reviewer import ExtractionReviewer, ReviewAction
from socialscikit.qualikit.segmenter import Segmenter
from socialscikit.qualikit.segment_extractor import ResearchQuestion, SegmentExtractor
from socialscikit.qualikit.theme_reviewer import ThemeReviewer

# ---------------------------------------------------------------------------
# Shared instances
# ---------------------------------------------------------------------------

_validator = DataValidator()
_deidentifier = Deidentifier()
_deident_reviewer = DeidentReviewer()
_theme_definer = ThemeDefiner()
_theme_reviewer = ThemeReviewer()
_confidence_ranker = ConfidenceRanker()
_coding_reviewer = CodingReviewer()
_exporter = Exporter()
_segmenter = Segmenter()
_extraction_reviewer = ExtractionReviewer()

# Disclaimer text
_DISCLAIMER = (
    "**重要提示：** 自动脱敏为初步处理工具。提交 IRB 审核前必须进行人工复核。"
    "本工具不保证完全去除所有身份信息。"
)


# ---------------------------------------------------------------------------
# Step 1: Upload & Validate
# ---------------------------------------------------------------------------


def _load_and_validate(file):
    if file is None:
        return None, "请上传数据文件。", None, None
    try:
        df = load_file(file.name)
    except DataLoadError as e:
        return None, f"加载失败：{e}", None, None

    report = _validator.validate(df, mode="qualikit")
    issues_rows = []
    for issue in report.issues:
        issues_rows.append({
            "级别": {"error": "❌ 错误", "warning": "⚠️ 警告", "info": "ℹ️ 信息"}[issue.severity],
            "位置": issue.field,
            "说明": issue.message,
            "可自动修复": "是" if issue.auto_fix_available else "否",
        })
    issues_df = pd.DataFrame(issues_rows) if issues_rows else None
    file_size = os.path.getsize(file.name) / 1024
    diag = generate_diagnostics(df, text_col=report.suggested_text_col, file_size_kb=file_size)
    summary = report.summary + "\n\n" + format_report(diag)
    return df, summary, issues_df, df.head(10)


def _apply_fixes(df_state):
    if df_state is None:
        return None, "没有数据可修复。"
    report = _validator.validate(df_state, mode="qualikit")
    fixed = apply_auto_fixes(df_state, report)
    new_report = _validator.validate(fixed, mode="qualikit")
    return fixed, f"已修复。{new_report.summary}（剩余 {len(fixed)} 行）"


# ---------------------------------------------------------------------------
# Step 2: De-identification
# ---------------------------------------------------------------------------


def _run_deident(df_state, text_col, entities_str, strategy):
    if df_state is None:
        return None, "请先上传数据。", None, None
    text_col = text_col.strip() or "text"
    if text_col not in df_state.columns:
        return None, f"未找到列 '{text_col}'。", None, None

    texts = df_state[text_col].dropna().astype(str).tolist()
    entities = [e.strip() for e in entities_str.split(",") if e.strip()]

    result = _deidentifier.process(texts, entities=entities or None, replacement_strategy=strategy)

    # Format log for display
    log_rows = Deidentifier.format_log_table(result.replacement_log)
    log_df = pd.DataFrame(log_rows) if log_rows else None

    stats_text = "检出实体：" + ", ".join(f"{k}: {v}" for k, v in result.coverage_stats.items())
    if not result.coverage_stats:
        stats_text = "未检出任何 PII 实体。"

    # Create review session
    session = _deident_reviewer.create_session(result, texts)

    return session, stats_text, log_df, result


def _deident_accept_all(session_state):
    if session_state is None:
        return None, "请先运行脱敏。"
    count = _deident_reviewer.accept_all(session_state)
    return session_state, f"已接受全部 {count} 项替换。"


def _deident_accept_high(session_state):
    if session_state is None:
        return None, "请先运行脱敏。"
    count = _deident_reviewer.accept_high_confidence(session_state, threshold=0.9)
    return session_state, f"已接受 {count} 项高置信度替换（>0.9）。"


# --- Per-item deident review ---

def _deident_review_table(session_state):
    """Build a review table from the deident session — one row per replacement."""
    if session_state is None:
        return None
    _status = {
        "pending": "⏳待审", "accepted": "✅接受",
        "edited": "✏️编辑", "rejected": "❌拒绝",
    }
    rows = []
    for i, item in enumerate(session_state.items):
        rows.append({
            "序号": i,
            "段落": item.record.text_id + 1,
            "原文片段": item.record.original_span,
            "替换为": item.final_replacement,
            "类型": item.record.entity_type,
            "置信度": f"{item.record.confidence:.2f}",
            "状态": _status.get(item.action.value, item.action.value),
        })
    return pd.DataFrame(rows) if rows else None


def _deident_stats_text(session_state):
    """Format deident review progress."""
    if session_state is None:
        return ""
    stats = _deident_reviewer.stats(session_state)
    total = sum(stats.values())
    reviewed = total - stats.get("pending", 0)
    pct = int(100 * reviewed / total) if total else 0
    return (
        f"总计: {total} | "
        f"✅接受: {stats.get('accepted', 0)} | "
        f"✏️编辑: {stats.get('edited', 0)} | "
        f"❌拒绝: {stats.get('rejected', 0)} | "
        f"⏳待审: {stats.get('pending', 0)} | "
        f"进度: {pct}%"
    )


def _deident_show_detail(session_state, index, segments):
    """Show detail HTML for a specific deident replacement."""
    import html as _esc
    if session_state is None:
        return '<p style="color:#888;">请先运行脱敏。</p>'
    try:
        idx = int(index)
        item = session_state.items[idx]
    except (ValueError, IndexError):
        return '<p style="color:#888;">请输入有效的序号。</p>'

    _status = {
        "pending": "⏳待审", "accepted": "✅接受",
        "edited": "✏️编辑", "rejected": "❌拒绝",
    }
    status = _status.get(item.action.value, item.action.value)
    text_id = item.record.text_id
    start, end = item.record.position

    # Get full segment text and highlight the span
    seg_text = ""
    if segments and text_id < len(segments):
        seg_text = segments[text_id].text

    if seg_text:
        before = _esc.escape(seg_text[:start])
        target = _esc.escape(seg_text[start:end])
        after = _esc.escape(seg_text[end:])
        highlighted = (
            f'{before}<mark style="background:#FFDEDE;padding:2px 0;'
            f'border-bottom:2px solid #E05555;">{target}</mark>{after}'
        )
    else:
        highlighted = f'<em style="color:#888;">无法获取段落文本</em>'

    return (
        f'<div style="padding:1rem;background:#fff;border:1px solid #e5e7eb;'
        f'border-radius:8px;font-family:Inter,sans-serif;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">'
        f'<span style="font-weight:600;color:#c0392b;">'
        f'[{idx}] 段落 {text_id+1} · {_esc.escape(item.record.entity_type)}</span>'
        f'<span style="color:#888;font-size:0.85rem;">{status} · 置信度 {item.record.confidence:.0%}</span>'
        f'</div>'
        f'<div style="font-size:0.88rem;margin-bottom:0.5rem;">'
        f'<b>原文：</b>{_esc.escape(item.record.original_span)}'
        f' → <b>替换为：</b>{_esc.escape(item.final_replacement)}</div>'
        f'<div style="white-space:pre-wrap;line-height:1.8;font-size:0.9rem;'
        f'color:#333;background:#fafafa;padding:0.75rem;border-radius:4px;">'
        f'{highlighted}</div></div>'
    )


def _deident_accept_one(session_state, index):
    """Accept a single deident replacement."""
    if session_state is None:
        return session_state, None, ""
    try:
        _deident_reviewer.accept(session_state, int(index))
        return session_state, _deident_review_table(session_state), _deident_stats_text(session_state)
    except (ValueError, IndexError) as e:
        return session_state, _deident_review_table(session_state), f"操作失败：{e}"


def _deident_reject_one(session_state, index):
    """Reject a single deident replacement (restore original)."""
    if session_state is None:
        return session_state, None, ""
    try:
        _deident_reviewer.reject(session_state, int(index))
        return session_state, _deident_review_table(session_state), _deident_stats_text(session_state)
    except (ValueError, IndexError) as e:
        return session_state, _deident_review_table(session_state), f"操作失败：{e}"


def _deident_edit_one(session_state, index, new_text):
    """Edit a deident replacement with custom text."""
    if session_state is None:
        return session_state, None, ""
    try:
        if not new_text or not new_text.strip():
            return session_state, _deident_review_table(session_state), "请输入替换文本。"
        _deident_reviewer.edit(session_state, int(index), new_text.strip())
        return session_state, _deident_review_table(session_state), _deident_stats_text(session_state)
    except (ValueError, IndexError) as e:
        return session_state, _deident_review_table(session_state), f"操作失败：{e}"


def _deident_accept_all_v2(session_state, lang="zh"):
    """Accept all pending, return updated table + stats."""
    if session_state is None:
        return session_state, None, t("msg.run_deident_first", lang)
    count = _deident_reviewer.accept_all(session_state)
    return session_state, _deident_review_table(session_state), t("msg.accepted_n", lang).format(count) + "\n" + _deident_stats_text(session_state)


def _deident_accept_high_v2(session_state, lang="zh"):
    """Accept high confidence, return updated table + stats."""
    if session_state is None:
        return session_state, None, t("msg.run_deident_first", lang)
    count = _deident_reviewer.accept_high_confidence(session_state, threshold=0.9)
    return session_state, _deident_review_table(session_state), t("msg.accepted_high_n", lang).format(count) + "\n" + _deident_stats_text(session_state)


def _deident_segments_v2(segments, entities_str, strategy, lang="zh"):
    """Run deident on segments, return (session, stats, review_table, progress, result)."""
    if not segments:
        return None, t("msg.segment_first", lang), None, "", None
    texts = [seg.text for seg in segments]
    entities = [e.strip() for e in entities_str.split(",") if e.strip()]
    result = _deidentifier.process(
        texts, entities=entities or None, replacement_strategy=strategy,
    )
    log_rows = Deidentifier.format_log_table(result.replacement_log)
    log_df = pd.DataFrame(log_rows) if log_rows else None
    entity_summary = ", ".join(
        f"{k}: {v}" for k, v in result.coverage_stats.items()
    )
    stats_text = t("msg.detected_entities", lang).format(entity_summary)
    if not result.coverage_stats:
        stats_text = t("msg.no_pii", lang)
    session = _deident_reviewer.create_session(result, texts)
    review_tbl = _deident_review_table(session)
    progress = _deident_stats_text(session)
    return session, stats_text, review_tbl, progress, result


def _deident_apply(session_state, df_state, text_col):
    if session_state is None or df_state is None:
        return df_state, "无法应用。"
    text_col = text_col.strip() or "text"
    final_texts = _deident_reviewer.apply(session_state)
    df_copy = df_state.copy()
    # Only update rows that had text
    valid_idx = df_copy[text_col].dropna().index[:len(final_texts)]
    for i, idx in enumerate(valid_idx):
        df_copy.at[idx, text_col] = final_texts[i]
    return df_copy, f"已应用脱敏，{len(final_texts)} 条文本已更新。"


# ---------------------------------------------------------------------------
# Step 3: Theme Definition
# ---------------------------------------------------------------------------


def _suggest_themes(df_state, text_col, n_themes, method, backend, model, api_key):
    if df_state is None:
        return None, "请先上传数据。", None
    text_col = text_col.strip() or "text"
    if text_col not in df_state.columns:
        return None, f"未找到列 '{text_col}'。", None

    texts = df_state[text_col].dropna().astype(str).tolist()

    llm = None
    if method == "llm":
        if not api_key and backend != "ollama":
            return None, "LLM 方法需要 API Key。", None
        llm = LLMClient(backend=backend, model=model, api_key=api_key or None)

    definer = ThemeDefiner(llm_client=llm)
    suggestions = definer.suggest_themes(texts, n_themes=int(n_themes), method=method)

    session = _theme_reviewer.create_session(suggestions)

    # Format summary
    lines = [f"建议 {len(suggestions)} 个主题："]
    for i, s in enumerate(suggestions):
        lines.append(f"  {i+1}. {s.name}（覆盖率 {s.estimated_coverage:.1%}）")
        lines.append(f"     {s.description}")
    summary = "\n".join(lines)

    # Themes table
    theme_rows = []
    for i, t in enumerate(session.themes):
        theme_rows.append({
            "序号": i + 1,
            "主题名称": t.name,
            "描述": t.description,
            "包含示例": "; ".join(t.inclusion_examples[:2]),
            "排除示例": "; ".join(t.exclusion_examples[:2]) if t.exclusion_examples else "（待填写）",
        })
    theme_df = pd.DataFrame(theme_rows)

    return session, summary, theme_df


def _check_theme_overlap(session_state):
    if session_state is None:
        return "请先生成主题。"
    definer = ThemeDefiner()
    warnings = definer.assess_overlap(session_state.themes)
    if not warnings:
        return "各主题之间语义区分度良好，无明显重叠。"
    return "\n".join(w["message"] for w in warnings)


def _lock_themes(session_state):
    if session_state is None:
        return None, "请先生成主题。"
    warnings = _theme_reviewer.lock(session_state)
    if session_state.locked:
        msg = "主题框架已锁定，可以开始编码。"
        if warnings:
            msg += "\n\n提示：\n" + "\n".join(f"  - {w}" for w in warnings)
        return session_state, msg
    return session_state, "锁定失败：\n" + "\n".join(f"  - {w}" for w in warnings)


# ---------------------------------------------------------------------------
# Step 4: LLM Coding
# ---------------------------------------------------------------------------


def _run_coding(df_state, text_col, theme_session_state, backend, model, api_key):
    if df_state is None:
        return None, None, "请先上传数据。", None
    if theme_session_state is None or not theme_session_state.locked:
        return None, None, "请先锁定主题框架。", None

    text_col = text_col.strip() or "text"
    texts = df_state[text_col].dropna().astype(str).tolist()
    themes = theme_session_state.themes

    if not api_key and backend != "ollama":
        return None, None, "请输入 API Key。", None

    llm = LLMClient(backend=backend, model=model, api_key=api_key or None)
    coder = Coder(llm)
    report = coder.code(texts, themes)

    # Rank by confidence
    ranked = _confidence_ranker.rank(report.results)
    summary = _confidence_ranker.summary(ranked)
    summary += f"\n\n编码完成：{report.n_coded}/{report.n_total} 条成功"

    # Create review session
    review_session = _coding_reviewer.create_session(ranked)

    # Results table
    rows = []
    for r in report.results:
        rows.append({
            "ID": r.text_id,
            "文本": r.text[:100] + "..." if len(r.text) > 100 else r.text,
            "主题": ", ".join(r.themes),
            "置信度": r.confidence_tier,
            "触发词": "; ".join(f"{k}: {', '.join(v)}" for k, v in r.trigger_words.items()),
        })
    results_df = pd.DataFrame(rows) if rows else None

    return report.results, review_session, summary, results_df


def _accept_all_high_coding(review_session_state):
    if review_session_state is None:
        return None, "请先运行编码。"
    count = _coding_reviewer.accept_all_high(review_session_state)
    stats = _coding_reviewer.stats(review_session_state)
    msg = f"已接受 {count} 条高置信度结果。\n"
    msg += f"高：{stats['high']['reviewed']}/{stats['high']['total']}，"
    msg += f"中：{stats['medium']['reviewed']}/{stats['medium']['total']}，"
    msg += f"低：{stats['low']['reviewed']}/{stats['low']['total']}"
    return review_session_state, msg


# ---------------------------------------------------------------------------
# Step 5: Export
# ---------------------------------------------------------------------------


def _export_all(coding_results_state, theme_session_state, review_session_state):
    if coding_results_state is None or theme_session_state is None:
        return None, None, "请先完成编码。"

    themes = theme_session_state.themes

    # Build review action map
    review_actions = {}
    if review_session_state is not None:
        for tier in [review_session_state.high, review_session_state.medium, review_session_state.low]:
            for item in tier:
                review_actions[item.result.text_id] = item.action.value

    bundle = _exporter.export(
        results=coding_results_state,
        themes=themes,
        review_actions=review_actions,
    )

    # Save files
    excel_path = _exporter.save_excel(bundle)
    memo_path = _exporter.save_memo(bundle)

    return excel_path, memo_path, f"已导出：\n- Excel: {excel_path}\n- 备忘录: {memo_path}"


# ---------------------------------------------------------------------------
# Phase 1 (Step 0): Raw text → structured extraction
# ---------------------------------------------------------------------------


def _get_interview_template():
    """Return path to the built-in interview example .txt file."""
    import gradio as gr
    path = str(Path(__file__).parent.parent / "core" / "templates" / "interview_example.txt")
    return gr.update(visible=True, value=path)


def _load_raw_text(file, lang="zh"):
    """Load a plain text file — returns (raw, preview, summary)."""
    if file is None:
        return "", "", t("msg.upload_first", lang)
    try:
        raw = Path(file.name).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw = Path(file.name).read_text(encoding="gb18030")
        except Exception as e:
            return "", "", t("msg.operation_failed", lang).format(e)
    except Exception as e:
        return "", "", t("msg.operation_failed", lang).format(e)
    n_chars = len(raw)
    n_lines = raw.count("\n") + 1
    preview = raw[:3000] + ("……（已截断）" if len(raw) > 3000 else "")
    return raw, preview, t("msg.loaded_chars", lang).format(n_chars, n_lines)


def _parse_rqs(rqs_text: str):
    """Parse RQs and optional sub-themes from structured input.

    Format::

        RQ1：社区公共服务体验
          医疗服务
          教育设施
        RQ2：经济与就业状况
          收入变化
          就业困难

    Non-indented lines with colon → RQ definitions.
    Indented lines below → sub-themes under the current RQ.
    """
    rqs: list[ResearchQuestion] = []
    current_rq: ResearchQuestion | None = None

    for line in rqs_text.splitlines():
        if not line.strip():
            continue

        # Indented → sub-theme under current RQ
        if line.startswith((" ", "\t")) and current_rq is not None:
            theme = line.strip().lstrip("-·•– ").strip()
            if theme:
                current_rq.sub_themes.append(theme)
        else:
            # Non-indented → RQ definition
            stripped = line.strip()
            for sep in ("：", ":"):
                if sep in stripped:
                    rq_id, desc = stripped.split(sep, 1)
                    current_rq = ResearchQuestion(
                        rq_id=rq_id.strip(), description=desc.strip(),
                    )
                    rqs.append(current_rq)
                    break
            else:
                current_rq = ResearchQuestion(
                    rq_id=f"RQ{len(rqs)+1}", description=stripped,
                )
                rqs.append(current_rq)
    return rqs


def _segment_and_preview(raw_text, mode, context_window, lang="zh"):
    """Segment text and return (segments_list, preview_df, summary)."""
    if not raw_text or not raw_text.strip():
        return [], None, t("msg.upload_first", lang)
    cw = int(context_window) if mode == "context_window" else 2
    segments = _segmenter.segment(raw_text, mode=mode, context_window=cw)
    if not segments:
        return [], None, t("msg.no_content", lang)
    rows = []
    for seg in segments:
        row = {
            "ID": seg.segment_id,
            "文本": seg.text[:120] + ("…" if len(seg.text) > 120 else ""),
            "行号": f"L{seg.position.line_start}–{seg.position.line_end}",
            "字符": f"{seg.position.char_start}–{seg.position.char_end}",
        }
        if mode == "context_window" and seg.core_sentence:
            row["核心句"] = seg.core_sentence[:60] + ("…" if len(seg.core_sentence) > 60 else "")
        rows.append(row)
    return segments, pd.DataFrame(rows), t("msg.segment_done", lang).format(len(segments), mode)


def _build_review_rows(session):
    """Build compact display rows — full text shown in detail panel below."""
    if session is None:
        return []
    _status_map = {
        "pending": "⏳待审", "accepted": "✅通过",
        "edited": "✏️已编辑", "rejected": "❌拒绝",
    }
    rows = []
    for i, item in enumerate(session.items):
        rows.append({
            "序号": i,
            "段落ID": item.result.segment_id,
            "RQ": item.final_rq_label,
            "子主题": item.final_sub_theme,
            "置信度": f"{item.result.confidence:.2f}",
            "状态": _status_map.get(item.action.value, item.action.value),
        })
    return rows


def _ext_refresh_table(session):
    """Return updated review table DataFrame."""
    rows = _build_review_rows(session)
    return pd.DataFrame(rows) if rows else None


def _ext_stats_text(session):
    """Format review progress string."""
    if session is None:
        return ""
    stats = _extraction_reviewer.stats(session)
    return (
        f"总计: {stats['total']} | "
        f"✅通过: {stats['accepted']} | "
        f"✏️编辑: {stats['edited']} | "
        f"❌拒绝: {stats['rejected']} | "
        f"⏳待审: {stats['pending']} | "
        f"进度: {stats['progress_pct']}%"
    )


def _run_extraction(raw_text, segments, rqs_text, backend, model, api_key):
    """LLM extraction — returns (session, summary, ext_tbl, rev_tbl, stats)."""
    if not segments:
        return None, "请先在「上传与分段」中进行分段。", None, None, ""
    rqs = _parse_rqs(rqs_text)
    if not rqs:
        return None, "请输入至少一个研究问题（格式：RQ1：描述）。", None, None, ""
    if not api_key and backend != "ollama":
        return None, "请输入 API Key。", None, None, ""

    # Show parsed RQs for transparency
    rq_lines = []
    for rq in rqs:
        sub = f" → 子主题: {', '.join(rq.sub_themes)}" if rq.sub_themes else " → 子主题: LLM自动生成"
        rq_lines.append(f"  {rq.rq_id}: {rq.description}{sub}")
    rq_info = "\n".join(rq_lines)

    llm = LLMClient(backend=backend, model=model, api_key=api_key or None)
    extractor = SegmentExtractor(llm)
    report = extractor.extract(segments, rqs)
    session = _extraction_reviewer.create_session(report, raw_text, segments, rqs)

    table = _ext_refresh_table(session)
    dist = "，".join(f"{k}: {v}" for k, v in report.rq_distribution.items())
    summary = (
        f"已解析 {len(rqs)} 个研究问题：\n{rq_info}\n\n"
        f"提取结果：{report.n_relevant} 个匹配 / {report.n_segments_total} 总段落\n"
        f"各 RQ 分布：{dist}"
    )
    return session, summary, table, table, _ext_stats_text(session)


def _ext_accept(session, index):
    """Accept extraction at given index."""
    if session is None:
        return session, None, "请先运行提取。"
    try:
        idx = int(index)
        _extraction_reviewer.accept(session, idx)
        return session, _ext_refresh_table(session), _ext_stats_text(session)
    except (ValueError, IndexError) as e:
        return session, _ext_refresh_table(session), f"操作失败：{e}"


def _ext_reject(session, index):
    """Reject extraction at given index."""
    if session is None:
        return session, None, "请先运行提取。"
    try:
        idx = int(index)
        _extraction_reviewer.reject(session, idx)
        return session, _ext_refresh_table(session), _ext_stats_text(session)
    except (ValueError, IndexError) as e:
        return session, _ext_refresh_table(session), f"操作失败：{e}"


def _ext_edit(session, index, new_rq, new_sub):
    """Edit RQ label and/or sub-theme."""
    if session is None:
        return session, None, "请先运行提取。"
    try:
        idx = int(index)
        rq = new_rq.strip() or None
        sub = new_sub.strip() or None
        _extraction_reviewer.edit(session, idx, new_rq_label=rq, new_sub_theme=sub)
        return session, _ext_refresh_table(session), _ext_stats_text(session)
    except (ValueError, IndexError) as e:
        return session, _ext_refresh_table(session), f"操作失败：{e}"


def _ext_accept_all_high(session, threshold):
    """Bulk-accept all items above confidence threshold."""
    if session is None:
        return session, None, "请先运行提取。"
    count = _extraction_reviewer.accept_all_high(session, threshold=float(threshold))
    msg = f"已批量接受 {count} 条（阈值 ≥ {threshold:.0%}）\n" + _ext_stats_text(session)
    return session, _ext_refresh_table(session), msg


def _ext_add_manual(session, seg_id, rq_label, sub_theme):
    """Manually add a segment the LLM missed."""
    if session is None:
        return session, None, "请先运行提取。"
    try:
        result = _extraction_reviewer.add_manual(
            session, int(seg_id), rq_label.strip(), sub_theme.strip(),
        )
        if result is None:
            return session, _ext_refresh_table(session), f"未找到段落 ID {int(seg_id)}，请查看分段预览表的 ID 列。"
        return session, _ext_refresh_table(session), _ext_stats_text(session)
    except Exception as e:
        return session, _ext_refresh_table(session), f"添加失败：{e}"


def _ext_show_context(session, index, raw_text):
    """Show full segment text + highlighted context in original document."""
    if session is None or not raw_text:
        return '<p style="color:#888;padding:1rem;">无可显示内容。请先运行提取。</p>'
    try:
        idx = int(index)
        item = session.items[idx]
    except (ValueError, IndexError):
        return '<p style="color:#888;padding:1rem;">请输入有效的序号。</p>'

    pos = item.result.position
    esc = _html_escape.escape
    status_map = {
        "pending": "⏳待审", "accepted": "✅通过",
        "edited": "✏️已编辑", "rejected": "❌拒绝",
    }
    status = status_map.get(item.action.value, item.action.value)

    # --- Part 1: full segment text ---
    full_text_html = (
        '<div style="font-family:Inter,sans-serif;padding:1rem;background:#fff;'
        'border:1px solid #e5e7eb;border-radius:8px;margin-bottom:0.75rem;">'
        '<div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">'
        f'<span style="font-weight:600;color:#4A90D9;">'
        f'[{idx}] 段落 {item.result.segment_id} · {esc(item.final_rq_label)}'
        f' · {esc(item.final_sub_theme)}</span>'
        f'<span style="color:#888;font-size:0.85rem;">'
        f'{status} · 置信度 {item.result.confidence:.0%}</span>'
        '</div>'
        f'<div style="white-space:pre-wrap;line-height:1.8;font-size:0.92rem;'
        f'color:#333;background:#fafafa;padding:0.75rem;border-radius:4px;">'
        f'{esc(item.result.text)}</div>'
        f'<div style="margin-top:0.5rem;font-size:0.85rem;color:#666;">'
        f'判断依据：{esc(item.result.reasoning)}</div>'
        '</div>'
    )

    # --- Part 2: highlighted context in original text ---
    ctx_html = ""
    if pos is not None:
        cs, ce = pos.char_start, pos.char_end
        ctx_start = max(0, cs - 300)
        ctx_end = min(len(raw_text), ce + 300)
        before = esc(raw_text[ctx_start:cs])
        target = esc(raw_text[cs:ce])
        after = esc(raw_text[ce:ctx_end])
        ctx_html = (
            '<div style="font-family:Inter,sans-serif;padding:1rem;background:#fafafa;'
            'border:1px solid #e5e7eb;border-radius:8px;line-height:1.8;'
            'font-size:0.88rem;color:#555;max-height:320px;overflow-y:auto;">'
            f'<div style="font-size:0.82rem;color:#888;margin-bottom:0.3rem;">'
            f'原文定位 L{pos.line_start}–{pos.line_end}</div>'
            '<div style="white-space:pre-wrap;">'
            f'{"…" if ctx_start > 0 else ""}'
            f'{before}'
            '<mark style="background:#FFF3CD;padding:2px 0;'
            'border-bottom:2px solid #E6A817;">'
            f'{target}</mark>'
            f'{after}'
            f'{"…" if ctx_end < len(raw_text) else ""}'
            '</div></div>'
        )

    return full_text_html + ctx_html


def _ext_export_excel(session, lang="zh"):
    """Export reviewed results to Excel."""
    if session is None:
        return None, t("msg.run_extraction_first", lang)
    df = _extraction_reviewer.export_to_dataframe(session)
    if df.empty:
        return None, t("msg.no_data", lang)
    path = os.path.join(tempfile.gettempdir(), "extraction_results.xlsx")
    df.to_excel(path, index=False)
    return path, t("msg.export_success", lang).format(len(df))


# ---------------------------------------------------------------------------
# Steps 2-3: Segment deident + Research Framework
# ---------------------------------------------------------------------------


def _deident_segments(segments, entities_str, strategy):
    """Run deidentification on segment texts (not CSV text column).

    Returns (session, stats_text, log_df, result).
    """
    if not segments:
        return None, "请先进行分段。", None, None

    texts = [seg.text for seg in segments]
    entities = [e.strip() for e in entities_str.split(",") if e.strip()]

    result = _deidentifier.process(
        texts, entities=entities or None, replacement_strategy=strategy,
    )

    log_rows = Deidentifier.format_log_table(result.replacement_log)
    log_df = pd.DataFrame(log_rows) if log_rows else None

    stats_text = "检出实体：" + ", ".join(
        f"{k}: {v}" for k, v in result.coverage_stats.items()
    )
    if not result.coverage_stats:
        stats_text = "未检出任何 PII 实体。"

    session = _deident_reviewer.create_session(result, texts)
    return session, stats_text, log_df, result


def _deident_apply_to_segments(deident_session, segments, lang="zh"):
    """Apply reviewed deidentification back to segments.

    Returns (updated_segments, message_str).
    """
    from socialscikit.qualikit.segmenter import TextSegment

    if deident_session is None or not segments:
        return segments, t("msg.cannot_apply", lang)

    final_texts = _deident_reviewer.apply(deident_session)

    updated = []
    for i, seg in enumerate(segments):
        new_text = final_texts[i] if i < len(final_texts) else seg.text
        updated.append(
            TextSegment(
                segment_id=seg.segment_id,
                text=new_text,
                position=seg.position,
                core_sentence=seg.core_sentence,
                core_char_start=seg.core_char_start,
                core_char_end=seg.core_char_end,
            )
        )

    return updated, t("msg.applied_deident", lang).format(len(final_texts))


def _confirm_rq_framework(rq_df, sub_theme_df, lang="zh"):
    """Parse RQs and sub-themes from two interactive DataFrames.

    Returns (rqs_list, summary_text, rq_choices_update, sub_choices_update).
    """
    import gradio as gr

    rqs: list[ResearchQuestion] = []

    if rq_df is None or rq_df.empty:
        return [], t("msg.at_least_one_rq", lang), gr.update(), gr.update()

    # Build sub-theme lookup: rq_id -> [sub_theme_name, ...]
    sub_lookup: dict[str, list[str]] = {}
    if sub_theme_df is not None and not sub_theme_df.empty:
        for _, row in sub_theme_df.iterrows():
            parent_rq = str(row.iloc[0]).strip()
            name = str(row.iloc[1]).strip()
            if parent_rq and name:
                sub_lookup.setdefault(parent_rq, []).append(name)

    # Parse RQs
    for _, row in rq_df.iterrows():
        rq_id = str(row.iloc[0]).strip()
        desc = str(row.iloc[1]).strip()
        if not rq_id or not desc:
            continue
        subs = sub_lookup.get(rq_id, [])
        rqs.append(ResearchQuestion(rq_id=rq_id, description=desc, sub_themes=subs))

    if not rqs:
        return [], t("msg.at_least_one_rq", lang), gr.update(), gr.update()

    # Build summary
    lines = []
    for rq in rqs:
        sub_str = ", ".join(rq.sub_themes) if rq.sub_themes else t("msg.sub_themes_auto", lang)
        lines.append(f"  {rq.rq_id}: {rq.description}  →  子主题: {sub_str}")
    summary = t("msg.framework_confirmed", lang).format(len(rqs)) + "\n" + "\n".join(lines)

    rq_choices = gr.update(choices=[rq.rq_id for rq in rqs])

    all_subs = []
    for rq in rqs:
        all_subs.extend(rq.sub_themes)
    sub_choices = (
        gr.update(choices=all_subs) if all_subs
        else gr.update(choices=[t("msg.sub_themes_auto", lang)])
    )

    return rqs, summary, rq_choices, sub_choices


def _suggest_sub_themes(segments, rq_df, backend, model, api_key):
    """Use LLM to suggest sub-themes for each RQ based on text segments."""
    if not segments:
        return "请先进行分段。"

    # Parse RQs from the dataframe
    rqs: list[ResearchQuestion] = []
    if rq_df is not None and not rq_df.empty:
        for _, row in rq_df.iterrows():
            rq_id = str(row.iloc[0]).strip()
            desc = str(row.iloc[1]).strip()
            if rq_id and desc:
                rqs.append(ResearchQuestion(rq_id=rq_id, description=desc))

    if not rqs:
        return "请先在研究框架中定义至少一个研究问题。"

    if not api_key and backend != "ollama":
        return "请输入 API Key。"

    # Build prompt with sample text
    sample_texts = [seg.text for seg in segments[:20]]
    sample_block = "\n---\n".join(sample_texts)

    rq_block = "\n".join(
        f"- {rq.rq_id}: {rq.description}" for rq in rqs
    )

    prompt = (
        "以下是一组质性研究的文本段落（样本）：\n\n"
        f"{sample_block}\n\n"
        "研究问题如下：\n"
        f"{rq_block}\n\n"
        "请为每个研究问题建议 3-5 个子主题分类，以便后续编码。"
        "格式：每个 RQ 下列出子主题名称，每行一个。"
    )

    try:
        llm = LLMClient(backend=backend, model=model, api_key=api_key or None)
        response = llm.complete(
            prompt=prompt,
            system="你是一位质性研究方法专家。请基于文本数据为研究框架建议合适的子主题分类。",
            max_tokens=2048,
        )
        return (
            f"AI 子主题建议（供参考，请手动填入上方表格）：\n\n"
            f"{response.text}\n\n"
            f"---\n模型: {model} | tokens: {response.input_tokens}+{response.output_tokens}"
        )
    except Exception as e:
        return f"建议生成失败：{e}"


def _run_extraction_v2(raw_text, segments, rqs, backend, model, api_key, lang="zh"):
    """LLM extraction with pre-parsed RQs.

    Like _run_extraction but accepts a list of ResearchQuestion directly
    instead of a free-text string.

    Returns (session, summary, ext_tbl, rev_tbl, stats_text).
    """
    if not segments:
        return None, t("msg.segment_first", lang), None, None, ""
    if not rqs:
        return None, t("msg.define_rq_first", lang), None, None, ""
    if not api_key and backend != "ollama":
        return None, t("msg.enter_api_key", lang), None, None, ""

    # Show parsed RQs for transparency
    rq_lines = []
    for rq in rqs:
        sub = (
            f" → 子主题: {', '.join(rq.sub_themes)}"
            if rq.sub_themes
            else " → 子主题: LLM自动生成"
        )
        rq_lines.append(f"  {rq.rq_id}: {rq.description}{sub}")
    rq_info = "\n".join(rq_lines)

    llm = LLMClient(backend=backend, model=model, api_key=api_key or None)
    extractor = SegmentExtractor(llm)
    report = extractor.extract(segments, rqs)
    session = _extraction_reviewer.create_session(report, raw_text, segments, rqs)

    table = _ext_refresh_table(session)
    dist = "，".join(f"{k}: {v}" for k, v in report.rq_distribution.items())
    summary = (
        f"已解析 {len(rqs)} 个研究问题：\n{rq_info}\n\n"
        f"提取结果：{report.n_relevant} 个匹配 / {report.n_segments_total} 总段落\n"
        f"各 RQ 分布：{dist}"
    )
    return session, summary, table, table, _ext_stats_text(session)


def _preview_segment_by_id(segments, seg_id):
    """Show HTML preview of a segment's content for the manual-add section."""
    if not segments:
        return '<p style="color:#888;padding:1rem;">无分段数据。</p>'

    try:
        target_id = int(seg_id)
    except (TypeError, ValueError):
        return '<p style="color:#c00;padding:1rem;">请输入有效的段落 ID（数字）。</p>'

    esc = _html_escape.escape

    for seg in segments:
        if seg.segment_id == target_id:
            pos = seg.position
            char_count = len(seg.text)
            return (
                '<div style="font-family:Inter,sans-serif;padding:1rem;'
                'background:#fff;border:1px solid #e5e7eb;border-radius:8px;">'
                '<div style="display:flex;justify-content:space-between;'
                'margin-bottom:0.5rem;">'
                f'<span style="font-weight:600;color:#4A90D9;">'
                f'段落 ID: {target_id}</span>'
                f'<span style="color:#888;font-size:0.85rem;">'
                f'L{pos.line_start}–{pos.line_end} · {char_count} 字符</span>'
                '</div>'
                f'<div style="white-space:pre-wrap;line-height:1.8;'
                f'font-size:0.92rem;color:#333;background:#fafafa;'
                f'padding:0.75rem;border-radius:4px;">'
                f'{esc(seg.text)}</div>'
                '</div>'
            )

    available = [str(s.segment_id) for s in segments[:20]]
    hint = ", ".join(available)
    if len(segments) > 20:
        hint += f" ... (共 {len(segments)} 个)"
    return (
        f'<p style="color:#c00;padding:1rem;">'
        f'未找到段落 ID {target_id}。可用 ID: {esc(hint)}</p>'
    )


def _get_sub_themes_for_rq(rqs, rq_id):
    """Get sub-theme dropdown choices for a specific RQ (cascading dropdown)."""
    import gradio as gr

    if not rqs or not rq_id:
        return gr.update(choices=["（自动）"], value="（自动）")

    for rq in rqs:
        if rq.rq_id == rq_id:
            if rq.sub_themes:
                return gr.update(choices=rq.sub_themes, value=rq.sub_themes[0])
            return gr.update(choices=["（自动）"], value="（自动）")

    return gr.update(choices=["（自动）"], value="（自动）")


def _ext_bridge_to_pipeline(session):
    """Convert reviewed extractions to DataFrame for the downstream pipeline.

    Returns (df, msg, summary, preview, col_update) to also refresh Step 1.
    """
    import gradio as gr
    if session is None:
        return None, "请先完成提取和审阅。", gr.update(), gr.update(), gr.update()
    df = _extraction_reviewer.export_to_dataframe(session)
    if df.empty:
        return None, "无可导入数据。", gr.update(), gr.update(), gr.update()
    bridge_df = pd.DataFrame({"text": df["text"]})
    msg = f"已将 {len(bridge_df)} 条提取文本导入至下游流程。可在 Step 1–5 中继续处理。"
    summary = f"从 Step 0 导入 {len(bridge_df)} 条文本。文本列: text"
    preview = bridge_df.head(10)
    col_update = gr.update(choices=list(bridge_df.columns), value="text")
    return bridge_df, msg, summary, preview, col_update


# ---------------------------------------------------------------------------
# Build Gradio App
# ---------------------------------------------------------------------------


def create_app() -> gr.Blocks:
    """Create the QualiKit Gradio application."""

    with gr.Blocks(
        title="SocialSciKit — QualiKit",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# SocialSciKit — QualiKit\n零代码质性编码工具，面向社会科学研究者。")

        # Shared state
        df_state = gr.State(None)
        deident_session_state = gr.State(None)
        deident_result_state = gr.State(None)
        theme_session_state = gr.State(None)
        coding_results_state = gr.State(None)
        coding_review_state = gr.State(None)

        # =============================================================
        # Tab 1: Upload
        # =============================================================
        with gr.Tab("1. 数据上传"):
            gr.Markdown("请上传 CSV 或 Excel 文件（访谈记录、开放式问卷等）。")
            with gr.Row():
                file_input = gr.File(label="上传数据文件", file_types=[".csv", ".xlsx", ".xls", ".json", ".jsonl", ".txt"])
                template_btn = gr.Button("下载模板")
                template_file = gr.File(label="模板文件", visible=False)

            summary_box = gr.Textbox(label="诊断报告", lines=12, interactive=False)
            issues_table = gr.Dataframe(label="问题列表", interactive=False)
            preview_table = gr.Dataframe(label="数据预览", interactive=False)

            with gr.Row():
                auto_fix_btn = gr.Button("自动修复", variant="secondary")
                fix_status = gr.Textbox(label="修复状态", interactive=False)

            file_input.change(
                fn=_load_and_validate, inputs=[file_input],
                outputs=[df_state, summary_box, issues_table, preview_table],
            )
            auto_fix_btn.click(fn=_apply_fixes, inputs=[df_state], outputs=[df_state, fix_status])
            template_btn.click(
                fn=lambda: str(get_template_path("qualikit")),
                outputs=[template_file],
            )

        # =============================================================
        # Tab 2: De-identification
        # =============================================================
        with gr.Tab("2. 脱敏"):
            gr.Markdown(_DISCLAIMER)
            with gr.Row():
                deident_text_col = gr.Textbox(label="文本列名", value="text")
                deident_entities = gr.Textbox(
                    label="检测实体类型（逗号分隔）",
                    value="PERSON, ORG, LOCATION, DATE, PHONE, EMAIL, URL",
                )
                deident_strategy = gr.Dropdown(
                    choices=["placeholder", "category", "redact"],
                    value="placeholder", label="替换策略",
                )

            deident_btn = gr.Button("运行脱敏", variant="primary")
            deident_stats = gr.Textbox(label="检测结果", interactive=False)
            deident_log = gr.Dataframe(label="替换日志", interactive=False)

            with gr.Row():
                accept_all_btn = gr.Button("全部接受")
                accept_high_btn = gr.Button("仅接受高置信度（>0.9）")
                apply_btn = gr.Button("应用到数据", variant="primary")
            deident_msg = gr.Textbox(label="状态", interactive=False)

            deident_btn.click(
                fn=_run_deident,
                inputs=[df_state, deident_text_col, deident_entities, deident_strategy],
                outputs=[deident_session_state, deident_stats, deident_log, deident_result_state],
            )
            accept_all_btn.click(
                fn=_deident_accept_all, inputs=[deident_session_state],
                outputs=[deident_session_state, deident_msg],
            )
            accept_high_btn.click(
                fn=_deident_accept_high, inputs=[deident_session_state],
                outputs=[deident_session_state, deident_msg],
            )
            apply_btn.click(
                fn=_deident_apply, inputs=[deident_session_state, df_state, deident_text_col],
                outputs=[df_state, deident_msg],
            )

        # =============================================================
        # Tab 3: Theme Definition
        # =============================================================
        with gr.Tab("3. 主题定义"):
            gr.Markdown(
                "AI 辅助生成主题建议。审核编辑后锁定主题框架，再开始编码。\n"
                "每个主题建议填写排除示例（Dunivin 2024：排除示例显著提升编码准确率）。"
            )
            with gr.Row():
                theme_text_col = gr.Textbox(label="文本列名", value="text")
                theme_n = gr.Number(label="主题数量", value=6, precision=0)
                theme_method = gr.Dropdown(choices=["tfidf", "llm"], value="tfidf", label="方法")

            with gr.Row():
                theme_backend = gr.Dropdown(choices=["openai", "anthropic", "ollama"], value="openai", label="LLM 后端（LLM方法时需要）")
                theme_model = gr.Textbox(label="模型名称", value="gpt-4o-mini")
                theme_api_key = gr.Textbox(label="API Key", type="password")

            suggest_btn = gr.Button("生成主题建议", variant="primary")
            theme_summary = gr.Textbox(label="主题建议", lines=10, interactive=False)
            theme_table = gr.Dataframe(label="主题列表", interactive=False)

            with gr.Row():
                overlap_btn = gr.Button("检查主题重叠")
                lock_btn = gr.Button("锁定主题框架", variant="primary")
            overlap_msg = gr.Textbox(label="重叠分析", interactive=False)
            lock_msg = gr.Textbox(label="锁定状态", interactive=False)

            suggest_btn.click(
                fn=_suggest_themes,
                inputs=[df_state, theme_text_col, theme_n, theme_method,
                        theme_backend, theme_model, theme_api_key],
                outputs=[theme_session_state, theme_summary, theme_table],
            )
            overlap_btn.click(
                fn=_check_theme_overlap, inputs=[theme_session_state],
                outputs=[overlap_msg],
            )
            lock_btn.click(
                fn=_lock_themes, inputs=[theme_session_state],
                outputs=[theme_session_state, lock_msg],
            )

        # =============================================================
        # Tab 4: LLM Coding
        # =============================================================
        with gr.Tab("4. 编码"):
            gr.Markdown("使用 LLM 对文本进行主题编码。需要先锁定主题框架。")
            with gr.Row():
                code_text_col = gr.Textbox(label="文本列名", value="text")
                code_backend = gr.Dropdown(choices=["openai", "anthropic", "ollama"], value="openai", label="LLM 后端")
                code_model = gr.Textbox(label="模型名称", value="gpt-4o-mini")
                code_api_key = gr.Textbox(label="API Key", type="password")

            code_btn = gr.Button("开始编码", variant="primary")
            code_summary = gr.Textbox(label="编码结果", lines=8, interactive=False)
            code_results = gr.Dataframe(label="编码详情", interactive=False)

            with gr.Row():
                accept_high_code_btn = gr.Button("接受全部高置信度")
            code_review_msg = gr.Textbox(label="审核状态", interactive=False)

            code_btn.click(
                fn=_run_coding,
                inputs=[df_state, code_text_col, theme_session_state,
                        code_backend, code_model, code_api_key],
                outputs=[coding_results_state, coding_review_state, code_summary, code_results],
            )
            accept_high_code_btn.click(
                fn=_accept_all_high_coding, inputs=[coding_review_state],
                outputs=[coding_review_state, code_review_msg],
            )

        # =============================================================
        # Tab 5: Export
        # =============================================================
        with gr.Tab("5. 导出"):
            gr.Markdown(
                "导出编码结果：\n"
                "- **摘录表（Excel）**：主题 | 文本 | 来源 | 置信度 | 审核状态\n"
                "- **主题共现矩阵**：哪些主题经常同时出现\n"
                "- **分析备忘录（Markdown）**：频率统计 + 典型引用 + 研究者备注"
            )
            export_btn = gr.Button("导出全部", variant="primary")
            export_excel = gr.File(label="Excel 文件")
            export_memo = gr.File(label="分析备忘录")
            export_msg = gr.Textbox(label="导出状态", interactive=False)

            export_btn.click(
                fn=_export_all,
                inputs=[coding_results_state, theme_session_state, coding_review_state],
                outputs=[export_excel, export_memo, export_msg],
            )

    return app


# ---------------------------------------------------------------------------
# Launch helper
# ---------------------------------------------------------------------------


def launch(port: int = 7861, share: bool = False) -> None:
    """Launch the QualiKit Gradio app."""
    app = create_app()
    app.launch(server_port=port, share=share)
