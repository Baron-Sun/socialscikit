"""Theme definition — AI-assisted theme suggestion for qualitative coding.

Supports two methods:
1. **TF-IDF + clustering** (fast, no LLM required): K-Means on TF-IDF vectors
   with automatic keyword extraction per cluster.
2. **LLM-based** (requires LLM client): Sends representative text samples to
   the LLM and asks it to propose themes with definitions.

Each suggested theme includes: name, description, representative texts,
and estimated coverage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from socialscikit.core.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ThemeSuggestion:
    """A single suggested theme."""

    name: str
    description: str
    representative_texts: list[str] = field(default_factory=list)
    estimated_coverage: float = 0.0  # proportion of corpus


@dataclass
class Theme:
    """A confirmed theme for coding (post-review)."""

    name: str
    description: str
    inclusion_examples: list[str] = field(default_factory=list)
    exclusion_examples: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ThemeDefiner
# ---------------------------------------------------------------------------


class ThemeDefiner:
    """AI-assisted theme suggestion for qualitative analysis.

    Usage::

        definer = ThemeDefiner(llm_client=llm)
        suggestions = definer.suggest_themes(
            texts=["interview text 1", "interview text 2", ...],
            n_themes=6,
            method="tfidf",
        )
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client

    def suggest_themes(
        self,
        texts: list[str],
        n_themes: int = 6,
        method: str = "tfidf",
    ) -> list[ThemeSuggestion]:
        """Suggest themes from a corpus of texts.

        Parameters
        ----------
        texts : list[str]
            Input texts (interview segments, open-ended responses, etc.).
        n_themes : int
            Target number of themes to suggest.
        method : str
            ``"tfidf"`` — TF-IDF clustering (fast, no LLM)
            ``"llm"`` — LLM-based theme suggestion (requires llm_client)

        Returns
        -------
        list[ThemeSuggestion]
        """
        if not texts:
            return []

        if method == "llm" and self.llm_client is not None:
            return self._suggest_llm(texts, n_themes)
        return self._suggest_tfidf(texts, n_themes)

    # ------------------------------------------------------------------
    # TF-IDF + K-Means clustering
    # ------------------------------------------------------------------

    def _suggest_tfidf(
        self, texts: list[str], n_themes: int,
    ) -> list[ThemeSuggestion]:
        """Cluster texts with TF-IDF + K-Means and extract themes."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        import numpy as np

        n_themes = min(n_themes, len(texts))
        if n_themes < 1:
            return []

        # Vectorize
        min_df = 1
        if len(texts) > 10:
            min_df = 2
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            max_df=0.95 if len(texts) > 5 else 1.0,
            min_df=min_df,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Cluster
        km = KMeans(n_clusters=n_themes, random_state=42, n_init=10)
        labels = km.fit_predict(tfidf_matrix)

        suggestions = []
        for cluster_id in range(n_themes):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_size = len(cluster_indices)

            if cluster_size == 0:
                continue

            # Top keywords for this cluster
            center = km.cluster_centers_[cluster_id]
            top_indices = center.argsort()[-8:][::-1]
            keywords = [feature_names[i] for i in top_indices if center[i] > 0]

            # Representative texts (closest to center)
            from sklearn.metrics.pairwise import cosine_similarity
            cluster_vectors = tfidf_matrix[cluster_indices]
            sims = cosine_similarity(cluster_vectors, center.reshape(1, -1)).flatten()
            top_text_indices = sims.argsort()[-min(5, cluster_size):][::-1]
            rep_texts = [texts[cluster_indices[i]] for i in top_text_indices]

            # Auto-generate name and description from keywords
            name = f"Theme: {', '.join(keywords[:3])}"
            description = f"Texts related to: {', '.join(keywords[:5])}"

            suggestions.append(ThemeSuggestion(
                name=name,
                description=description,
                representative_texts=rep_texts,
                estimated_coverage=round(cluster_size / len(texts), 3),
            ))

        # Sort by coverage descending
        suggestions.sort(key=lambda s: s.estimated_coverage, reverse=True)
        return suggestions

    # ------------------------------------------------------------------
    # LLM-based theme suggestion
    # ------------------------------------------------------------------

    def _suggest_llm(
        self, texts: list[str], n_themes: int,
    ) -> list[ThemeSuggestion]:
        """Use LLM to suggest themes from text samples."""
        import json

        # Sample texts (up to 30 to fit context window)
        sample = texts[:30] if len(texts) > 30 else texts
        texts_block = "\n---\n".join(f"[{i+1}] {t}" for i, t in enumerate(sample))

        prompt = f"""\
Analyze the following {len(sample)} text segments from qualitative research data.
Identify {n_themes} distinct themes that emerge from these texts.

For each theme, provide:
1. A concise theme name (2-5 words)
2. A clear description (1-2 sentences defining the theme)
3. The indices (1-based) of 2-3 representative texts from the samples

Return a JSON array of objects with keys: "name", "description", "text_indices"

Text segments:
{texts_block}

Return ONLY valid JSON (no markdown fencing):"""

        system = "You are an expert qualitative researcher helping identify themes in interview/survey data."

        resp = self.llm_client.complete(prompt, system=system, max_tokens=4096)

        try:
            parsed = json.loads(resp.text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM theme response, falling back to TF-IDF")
            return self._suggest_tfidf(texts, n_themes)

        suggestions = []
        for item in parsed:
            indices = item.get("text_indices", [])
            rep_texts = []
            for idx in indices:
                adj_idx = int(idx) - 1  # 1-based to 0-based
                if 0 <= adj_idx < len(sample):
                    rep_texts.append(sample[adj_idx])

            suggestions.append(ThemeSuggestion(
                name=item.get("name", "Unnamed Theme"),
                description=item.get("description", ""),
                representative_texts=rep_texts,
                estimated_coverage=round(len(rep_texts) / len(texts), 3),
            ))

        return suggestions

    # ------------------------------------------------------------------
    # Theme overlap assessment
    # ------------------------------------------------------------------

    def assess_overlap(
        self, themes: list[Theme],
    ) -> list[dict]:
        """Assess semantic overlap between theme definitions.

        Returns a list of overlap warnings for theme pairs with high similarity.
        """
        if len(themes) < 2:
            return []

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Combine name + description for each theme
        theme_texts = [f"{t.name}. {t.description}" for t in themes]
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(theme_texts)
        sim_matrix = cosine_similarity(tfidf)

        warnings = []
        for i in range(len(themes)):
            for j in range(i + 1, len(themes)):
                overlap = round(float(sim_matrix[i, j]) * 100, 1)
                if overlap > 40:
                    warnings.append({
                        "theme_a": themes[i].name,
                        "theme_b": themes[j].name,
                        "overlap_pct": overlap,
                        "message": (
                            f"主题 \"{themes[i].name}\" 与 \"{themes[j].name}\" "
                            f"语义重叠度 {overlap}%，建议进一步区分两者的边界，或考虑合并。"
                        ),
                    })

        return warnings
