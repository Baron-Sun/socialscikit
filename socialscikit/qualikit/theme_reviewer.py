"""Interactive theme review — accept, edit, delete, merge, and add themes.

Manages the user review loop for AI-suggested themes before coding begins.
Enforces the requirement that each theme must have at least 1 exclusion
example (Dunivin 2024: exclusion criteria significantly improve coding accuracy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from socialscikit.qualikit.theme_definer import Theme, ThemeSuggestion


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ThemeReviewSession:
    """Tracks the theme review state."""

    themes: list[Theme] = field(default_factory=list)
    locked: bool = False  # True once user confirms and locks themes


# ---------------------------------------------------------------------------
# ThemeReviewer
# ---------------------------------------------------------------------------


class ThemeReviewer:
    """Interactive theme review and editing.

    Usage::

        reviewer = ThemeReviewer()
        session = reviewer.create_session(suggestions)
        reviewer.edit_theme(session, 0, name="New Name")
        reviewer.add_theme(session, Theme(name="Custom", description="..."))
        reviewer.merge_themes(session, [0, 1], merged_name="Combined")
        reviewer.lock(session)  # finalize for coding
    """

    def create_session(
        self, suggestions: list[ThemeSuggestion],
    ) -> ThemeReviewSession:
        """Create a review session from theme suggestions.

        Converts ThemeSuggestion objects into editable Theme objects.
        """
        themes = []
        for s in suggestions:
            themes.append(Theme(
                name=s.name,
                description=s.description,
                inclusion_examples=list(s.representative_texts),
                exclusion_examples=[],
            ))
        return ThemeReviewSession(themes=themes)

    def accept_theme(self, session: ThemeReviewSession, index: int) -> Theme:
        """Accept a theme as-is (no-op, but explicit confirmation)."""
        if session.locked:
            raise RuntimeError("Session is locked. Cannot modify themes.")
        return session.themes[index]

    def edit_theme(
        self,
        session: ThemeReviewSession,
        index: int,
        name: str | None = None,
        description: str | None = None,
        inclusion_examples: list[str] | None = None,
        exclusion_examples: list[str] | None = None,
    ) -> Theme:
        """Edit a theme's properties."""
        if session.locked:
            raise RuntimeError("Session is locked. Cannot modify themes.")
        if index < 0 or index >= len(session.themes):
            raise IndexError(f"Index {index} out of range [0, {len(session.themes)}).")

        theme = session.themes[index]
        if name is not None:
            theme.name = name
        if description is not None:
            theme.description = description
        if inclusion_examples is not None:
            theme.inclusion_examples = inclusion_examples
        if exclusion_examples is not None:
            theme.exclusion_examples = exclusion_examples
        return theme

    def delete_theme(self, session: ThemeReviewSession, index: int) -> Theme:
        """Remove a theme from the session."""
        if session.locked:
            raise RuntimeError("Session is locked. Cannot modify themes.")
        if index < 0 or index >= len(session.themes):
            raise IndexError(f"Index {index} out of range [0, {len(session.themes)}).")
        return session.themes.pop(index)

    def add_theme(self, session: ThemeReviewSession, theme: Theme) -> int:
        """Add a new custom theme. Returns its index."""
        if session.locked:
            raise RuntimeError("Session is locked. Cannot modify themes.")
        session.themes.append(theme)
        return len(session.themes) - 1

    def merge_themes(
        self,
        session: ThemeReviewSession,
        indices: list[int],
        merged_name: str,
        merged_description: str = "",
    ) -> Theme:
        """Merge multiple themes into one.

        Parameters
        ----------
        indices : list[int]
            Indices of themes to merge (must be >= 2).
        merged_name : str
        merged_description : str
        """
        if session.locked:
            raise RuntimeError("Session is locked. Cannot modify themes.")
        if len(indices) < 2:
            raise ValueError("Need at least 2 themes to merge.")

        # Validate indices
        for idx in indices:
            if idx < 0 or idx >= len(session.themes):
                raise IndexError(f"Index {idx} out of range.")

        # Combine examples
        all_inclusion = []
        all_exclusion = []
        for idx in indices:
            all_inclusion.extend(session.themes[idx].inclusion_examples)
            all_exclusion.extend(session.themes[idx].exclusion_examples)

        merged = Theme(
            name=merged_name,
            description=merged_description,
            inclusion_examples=all_inclusion,
            exclusion_examples=all_exclusion,
        )

        # Remove old themes (in reverse order to preserve indices)
        for idx in sorted(indices, reverse=True):
            session.themes.pop(idx)

        session.themes.append(merged)
        return merged

    def validate_for_coding(self, session: ThemeReviewSession) -> list[str]:
        """Check if themes are ready for coding.

        Returns a list of warning messages. Empty list means all good.

        Dunivin (2024): exclusion criteria significantly improve coding accuracy.
        Each theme should have at least 1 exclusion example.
        """
        warnings = []
        if not session.themes:
            warnings.append("没有定义任何主题。")
            return warnings

        for i, theme in enumerate(session.themes):
            if not theme.name.strip():
                warnings.append(f"主题 {i+1}：名称不能为空。")
            if not theme.description.strip():
                warnings.append(f"主题 \"{theme.name}\"：描述不能为空。")
            if not theme.exclusion_examples:
                warnings.append(
                    f"主题 \"{theme.name}\"：缺少排除示例。"
                    f"排除示例帮助 AI 区分边界情况，显著提升编码准确率（Dunivin 2024）。"
                )

        return warnings

    def lock(self, session: ThemeReviewSession) -> list[str]:
        """Lock the theme framework for coding.

        Returns validation warnings. If there are errors (empty name/description),
        the session is NOT locked.
        """
        warnings = self.validate_for_coding(session)
        # Only block on empty names/descriptions; exclusion examples are warned but not blocking
        has_errors = any("不能为空" in w or "没有定义" in w for w in warnings)
        if not has_errors:
            session.locked = True
        return warnings

    def unlock(self, session: ThemeReviewSession) -> None:
        """Unlock the session for further editing."""
        session.locked = False

    def get_themes_for_coding(self, session: ThemeReviewSession) -> list[Theme]:
        """Return the finalized theme list for the coding step."""
        if not session.locked:
            raise RuntimeError("Session must be locked before coding. Call lock() first.")
        return list(session.themes)
