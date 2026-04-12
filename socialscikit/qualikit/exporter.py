"""Structured export — excerpts table, co-occurrence matrix, and memo draft.

Outputs:
1. Excerpts table (Excel): theme | text | source | text_id | confidence | review status
2. Theme co-occurrence matrix: heatmap data showing which themes co-occur
3. Analysis memo draft (Markdown): per-theme frequency stats + sample quotes + notes field
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from socialscikit.qualikit.coder import CodingResult
from socialscikit.qualikit.theme_definer import Theme


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExportBundle:
    """All export artifacts from a QualiKit session."""

    excerpts_df: pd.DataFrame
    cooccurrence_df: pd.DataFrame
    memo_text: str
    excel_path: str | None = None  # path to saved Excel file


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


class Exporter:
    """Export qualitative coding results in structured formats.

    Usage::

        exporter = Exporter()
        bundle = exporter.export(
            results=coding_results,
            themes=themes,
            review_actions=review_map,
        )
    """

    def export(
        self,
        results: list[CodingResult],
        themes: list[Theme],
        review_actions: dict[int, str] | None = None,
        source_label: str = "",
    ) -> ExportBundle:
        """Generate all export artifacts.

        Parameters
        ----------
        results : list[CodingResult]
        themes : list[Theme]
        review_actions : dict[int, str] or None
            Map of text_id -> review action string (e.g. "accepted", "rejected").
        source_label : str
            Label for the data source (e.g. filename).
        """
        excerpts = self.build_excerpts_table(results, review_actions, source_label)
        cooccurrence = self.build_cooccurrence_matrix(results, themes)
        memo = self.generate_memo(results, themes)

        return ExportBundle(
            excerpts_df=excerpts,
            cooccurrence_df=cooccurrence,
            memo_text=memo,
        )

    # ------------------------------------------------------------------
    # 1. Excerpts Table
    # ------------------------------------------------------------------

    def build_excerpts_table(
        self,
        results: list[CodingResult],
        review_actions: dict[int, str] | None = None,
        source_label: str = "",
    ) -> pd.DataFrame:
        """Build the excerpts table: one row per (text, theme) pair."""
        review_actions = review_actions or {}
        rows = []
        for result in results:
            for theme in result.themes:
                rows.append({
                    "主题": theme,
                    "文本段落": result.text,
                    "来源": source_label,
                    "段落ID": result.text_id,
                    "置信度": result.confidences.get(theme, 0.0),
                    "触发词": ", ".join(result.trigger_words.get(theme, [])),
                    "审核状态": review_actions.get(result.text_id, "未审核"),
                    "编码依据": result.reasoning,
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 2. Co-occurrence Matrix
    # ------------------------------------------------------------------

    def build_cooccurrence_matrix(
        self,
        results: list[CodingResult],
        themes: list[Theme],
    ) -> pd.DataFrame:
        """Build a theme co-occurrence matrix.

        Cell (i, j) = number of texts coded with both theme i and theme j.
        Diagonal = total count of each theme.
        """
        theme_names = [t.name for t in themes]
        matrix = pd.DataFrame(0, index=theme_names, columns=theme_names)

        for result in results:
            assigned = [t for t in result.themes if t in theme_names]
            for t in assigned:
                matrix.loc[t, t] += 1
            for i in range(len(assigned)):
                for j in range(i + 1, len(assigned)):
                    matrix.loc[assigned[i], assigned[j]] += 1
                    matrix.loc[assigned[j], assigned[i]] += 1

        return matrix

    # ------------------------------------------------------------------
    # 3. Analysis Memo
    # ------------------------------------------------------------------

    def generate_memo(
        self,
        results: list[CodingResult],
        themes: list[Theme],
    ) -> str:
        """Generate a Markdown analysis memo draft."""
        total_texts = len(results)
        theme_names = {t.name for t in themes}

        # Count theme occurrences
        theme_counts: dict[str, int] = Counter()
        theme_texts: dict[str, list[str]] = {t.name: [] for t in themes}

        for result in results:
            for t in result.themes:
                if t in theme_names:
                    theme_counts[t] += 1
                    if len(theme_texts[t]) < 3:
                        theme_texts[t].append(result.text)

        lines = [
            "# 质性编码分析备忘录",
            "",
            f"**总文本数：** {total_texts}",
            f"**主题数量：** {len(themes)}",
            f"**生成时间：** （自动生成）",
            "",
            "---",
            "",
        ]

        for theme in themes:
            count = theme_counts.get(theme.name, 0)
            pct = round(count / total_texts * 100, 1) if total_texts > 0 else 0
            lines.append(f"## {theme.name}")
            lines.append(f"")
            lines.append(f"**定义：** {theme.description}")
            lines.append(f"")
            lines.append(f"**频率：** {count} 条文本（{pct}%）")
            lines.append(f"")

            quotes = theme_texts.get(theme.name, [])
            if quotes:
                lines.append("**典型引用：**")
                for q in quotes:
                    # Truncate long quotes
                    display = q[:200] + "..." if len(q) > 200 else q
                    lines.append(f'> "{display}"')
                    lines.append("")

            lines.append("**研究者备注：**")
            lines.append("（在此添加您的分析笔记）")
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Save to Excel
    # ------------------------------------------------------------------

    def save_excel(
        self, bundle: ExportBundle, path: str | None = None,
    ) -> str:
        """Save excerpts and co-occurrence to an Excel file.

        Returns the file path.
        """
        if path is None:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".xlsx", delete=False, prefix="qualikit_export_",
            )
            path = tmp.name
            tmp.close()

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            bundle.excerpts_df.to_excel(writer, sheet_name="摘录表", index=False)
            bundle.cooccurrence_df.to_excel(writer, sheet_name="共现矩阵")

        bundle.excel_path = path
        return path

    def save_memo(self, bundle: ExportBundle, path: str | None = None) -> str:
        """Save the memo to a Markdown file. Returns the file path."""
        if path is None:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".md", delete=False, prefix="qualikit_memo_", mode="w",
            )
            path = tmp.name
            tmp.close()

        with open(path, "w", encoding="utf-8") as f:
            f.write(bundle.memo_text)

        return path
