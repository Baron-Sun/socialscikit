"""PII de-identification — detect and replace personal identifiers.

Technical approach: regex-based entity detection for common PII patterns
(email, phone, URL, SSN, IP address) plus optional spaCy NER for names,
organizations, locations, and dates.

Important disclaimer (must be surfaced in UI):
    Automated de-identification is a first-pass tool. Manual review is
    required before IRB submission. This tool does not guarantee complete
    removal of identifying information.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------

SUPPORTED_ENTITIES = [
    "PERSON", "ORG", "LOCATION", "DATE",
    "PHONE", "EMAIL", "URL", "SSN", "IP_ADDRESS", "ID_CARD",
]


class ReplacementStrategy(str, Enum):
    PLACEHOLDER = "placeholder"   # [NAME_1], [ORG_1]
    CATEGORY = "category"         # [PERSON], [ORGANIZATION]
    REDACT = "redact"             # [REDACTED]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ReplacementRecord:
    """A single replacement operation."""

    text_id: int
    original_span: str
    replacement: str
    entity_type: str
    confidence: float
    position: tuple[int, int]  # (start, end) in original text


@dataclass
class DeidentResult:
    """Result of de-identification processing."""

    deidentified_texts: list[str]
    replacement_log: list[ReplacementRecord]
    coverage_stats: dict[str, int]  # entity_type -> count


# ---------------------------------------------------------------------------
# Regex patterns for common PII
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ),
    "PHONE": re.compile(
        r'(?:'
        # Chinese mobile: 13x-19x, 11 digits, optional +86
        r'(?:\+?86[-\s]?)?1[3-9]\d{9}'
        r'|'
        # Chinese landline: area code (2-4 digits) + number (7-8 digits)
        r'(?:0\d{2,3}[-\s]?)?\d{7,8}'
        r'|'
        # International / US: optional country code + 10 digits
        r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}'
        r'(?:\s*(?:ext|x|extension)\s*\d{1,5})?'
        r')'
    ),
    "URL": re.compile(
        r'https?://[^\s<>"\']+|www\.[^\s<>"\']+',
    ),
    "SSN": re.compile(
        r'\b\d{3}-\d{2}-\d{4}\b'
    ),
    "IP_ADDRESS": re.compile(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ),
    # --- Chinese PII patterns ---
    #
    # Design rationale: regex-only approach (no spaCy dependency).
    # Pattern 1: surname + 0-2 chars + REQUIRED honorific → very precise.
    # Pattern 2: prefix (老/小/阿) + surname → precise.
    # Pattern 3: English multi-word capitalized names.
    # Bare "surname + name" without title is intentionally excluded to
    # avoid false positives (e.g. 方便, 高兴, 马上).
    "PERSON": re.compile(
        r'(?:'
        # P1: surname + optional given name (0-2 chars) + required title
        # e.g. 张女士, 李先生, 王小明老师, 陈教授
        r'(?:'
        r'[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜'
        r'戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐'
        r'费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄'
        r'和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁'
        r'杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁锺徐邱骆高夏蔡田樊胡凌霍'
        r'虞万支柯管卢莫经房裘干解应宗丁宣邓郁单杭洪包诸左石崔吉龚程邢裴陆'
        r'荣翁荀甄曲家封储靳段富巫焦牧山谷车侯全班秋仲宫仇栾甘戎祖武符刘景'
        r'詹束龙叶幸司韶黎薄印宿白怀蒲从索赖卓蔺屠蒙池乔阴胥苍双闻莘翟谭贡'
        r'劳姬申扶冉宰雍桑桂牛寿通边燕冀浦尚农温别庄晏柴瞿阎充慕连茹习艾鱼'
        r'容向古易慎戈廖庾居衡步都耿满弘匡国文寇广禄东欧沃越隆师巩聂勾敖融冷'
        r'辛阚那简饶空曾沙养须丰关查后荆红游竺权盖益桓公]'
        r'[\u4e00-\u9fff]{0,2}'
        r'(?:先生|女士|老师|教授|医生|博士|同志|大爷|大妈|阿姨|叔叔|大哥|大姐|小姐|同学|师傅)'
        r')'
        r'|'
        # P2: prefix + surname  (老张, 小王, 阿芳)
        r'(?:老|小|阿)[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜'
        r'戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐'
        r'费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄'
        r'和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁'
        r'杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁锺徐邱骆高夏蔡田樊胡凌霍'
        r'虞万支柯管卢莫经房裘干解应宗丁宣邓郁单杭洪包诸左石崔吉龚程邢裴陆'
        r'荣翁荀甄曲家封储靳段富巫焦牧山谷车侯全班秋仲宫仇栾甘戎祖武符刘景'
        r'詹束龙叶幸司韶黎薄印宿白怀蒲从索赖卓蔺屠蒙池乔阴胥苍双闻莘翟谭贡'
        r'劳姬申扶冉宰雍桑桂牛寿通边燕冀浦尚农温别庄晏柴瞿阎充慕连茹习艾鱼'
        r'容向古易慎戈廖庾居衡步都耿满弘匡国文寇广禄东欧沃越隆师巩聂勾敖融冷'
        r'辛阚那简饶空曾沙养须丰关查后荆红游竺权盖益桓公]'
        r'|'
        # P3: English full names (John Smith, Mary Jane Watson)
        r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)+'
        r')'
    ),
    "ID_CARD": re.compile(
        # Chinese ID card: 18 digits (last may be X)
        r'\b\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b'
    ),
}


# ---------------------------------------------------------------------------
# Deidentifier
# ---------------------------------------------------------------------------


class Deidentifier:
    """Detect and replace personally identifiable information in texts.

    Usage::

        deident = Deidentifier()
        result = deident.process(
            texts=["Dr. Sarah Chen works at Stanford."],
            entities=["PERSON", "ORG", "EMAIL"],
            replacement_strategy="placeholder",
        )
    """

    def __init__(self):
        self._nlp = None  # lazy-loaded spaCy model
        self._nlp_loaded = False

    def _load_spacy(self) -> bool:
        """Attempt to load spaCy NER model. Returns True if successful."""
        if self._nlp_loaded:
            return self._nlp is not None

        self._nlp_loaded = True
        try:
            import spacy
            # Try transformer model first, fall back to smaller models
            for model_name in ["en_core_web_trf", "en_core_web_lg", "en_core_web_sm"]:
                try:
                    self._nlp = spacy.load(model_name)
                    logger.info("Loaded spaCy model: %s", model_name)
                    return True
                except OSError:
                    continue
            logger.warning("No spaCy model found. NER-based detection disabled. "
                           "Install with: python -m spacy download en_core_web_sm")
            return False
        except ImportError:
            logger.warning("spaCy not installed. NER-based detection disabled.")
            return False

    def process(
        self,
        texts: list[str],
        entities: list[str] | None = None,
        replacement_strategy: str = "placeholder",
    ) -> DeidentResult:
        """De-identify a list of texts.

        Parameters
        ----------
        texts : list[str]
            Input texts to process.
        entities : list[str] or None
            Entity types to detect. Defaults to all supported types.
        replacement_strategy : str
            ``"placeholder"`` → [NAME_1], [ORG_1], etc.
            ``"category"`` → [PERSON], [ORGANIZATION], etc.
            ``"redact"`` → [REDACTED]

        Returns
        -------
        DeidentResult
        """
        if entities is None:
            entities = list(SUPPORTED_ENTITIES)

        strategy = ReplacementStrategy(replacement_strategy)

        # Counters for placeholder numbering
        entity_counters: dict[str, int] = {}
        # Map from (original_span, entity_type) to consistent replacement
        entity_map: dict[tuple[str, str], str] = {}

        all_records: list[ReplacementRecord] = []
        deidentified: list[str] = []
        coverage: dict[str, int] = {e: 0 for e in entities}

        for text_id, text in enumerate(texts):
            # Collect all detections
            detections: list[tuple[int, int, str, str, float]] = []

            # 1. Regex-based detection
            regex_entities = {"EMAIL", "PHONE", "URL", "SSN", "IP_ADDRESS",
                              "PERSON", "ID_CARD"}
            for ent_type in entities:
                if ent_type in regex_entities and ent_type in _PATTERNS:
                    for match in _PATTERNS[ent_type].finditer(text):
                        # PERSON regex: lower confidence (pattern-based)
                        conf = 0.88 if ent_type == "PERSON" else 0.95
                        detections.append((
                            match.start(), match.end(),
                            ent_type, match.group(), conf,
                        ))

            # 2. spaCy NER detection (for PERSON, ORG, LOCATION, DATE)
            ner_entities = {"PERSON", "ORG", "LOCATION", "DATE"}
            requested_ner = set(entities) & ner_entities
            if requested_ner and self._load_spacy():
                doc = self._nlp(text)
                spacy_to_our = {
                    "PERSON": "PERSON",
                    "ORG": "ORG",
                    "GPE": "LOCATION",
                    "LOC": "LOCATION",
                    "DATE": "DATE",
                    "FAC": "LOCATION",
                }
                for ent in doc.ents:
                    mapped = spacy_to_our.get(ent.label_)
                    if mapped and mapped in requested_ner:
                        detections.append((
                            ent.start_char, ent.end_char,
                            mapped, ent.text,
                            round(0.85 + 0.1 * min(len(ent.text.split()), 3) / 3, 2),
                        ))

            # Sort by position (start descending for safe replacement)
            detections.sort(key=lambda d: d[0])

            # De-duplicate overlapping detections (keep higher confidence)
            deduped: list[tuple[int, int, str, str, float]] = []
            for det in detections:
                start, end, etype, span, conf = det
                overlaps = False
                for existing in deduped:
                    e_start, e_end = existing[0], existing[1]
                    if start < e_end and end > e_start:
                        overlaps = True
                        break
                if not overlaps:
                    deduped.append(det)

            # Build replacement text (process from end to preserve positions)
            result_text = text
            records_for_text: list[ReplacementRecord] = []

            for start, end, etype, span, conf in sorted(deduped, key=lambda d: d[0], reverse=True):
                replacement = self._get_replacement(
                    span, etype, strategy, entity_counters, entity_map,
                )
                result_text = result_text[:start] + replacement + result_text[end:]

                coverage[etype] = coverage.get(etype, 0) + 1
                records_for_text.append(ReplacementRecord(
                    text_id=text_id,
                    original_span=span,
                    replacement=replacement,
                    entity_type=etype,
                    confidence=conf,
                    position=(start, end),
                ))

            deidentified.append(result_text)
            all_records.extend(reversed(records_for_text))

        # Filter out zero-count entities
        coverage = {k: v for k, v in coverage.items() if v > 0}

        return DeidentResult(
            deidentified_texts=deidentified,
            replacement_log=all_records,
            coverage_stats=coverage,
        )

    @staticmethod
    def _get_replacement(
        span: str,
        entity_type: str,
        strategy: ReplacementStrategy,
        counters: dict[str, int],
        entity_map: dict[tuple[str, str], str],
    ) -> str:
        """Generate a consistent replacement string for an entity."""
        key = (span.lower().strip(), entity_type)

        if key in entity_map:
            return entity_map[key]

        if strategy == ReplacementStrategy.REDACT:
            replacement = "[REDACTED]"
        elif strategy == ReplacementStrategy.CATEGORY:
            type_labels = {
                "PERSON": "PERSON", "ORG": "ORGANIZATION",
                "LOCATION": "LOCATION", "DATE": "DATE",
                "PHONE": "PHONE", "EMAIL": "EMAIL",
                "URL": "URL", "SSN": "SSN", "IP_ADDRESS": "IP_ADDRESS",
            }
            replacement = f"[{type_labels.get(entity_type, entity_type)}]"
        else:
            # Placeholder with numbering: [NAME_1], [ORG_2], etc.
            type_prefixes = {
                "PERSON": "NAME", "ORG": "ORG",
                "LOCATION": "LOCATION", "DATE": "DATE",
                "PHONE": "PHONE", "EMAIL": "EMAIL",
                "URL": "URL", "SSN": "SSN", "IP_ADDRESS": "IP",
            }
            prefix = type_prefixes.get(entity_type, entity_type)
            counters[entity_type] = counters.get(entity_type, 0) + 1
            replacement = f"[{prefix}_{counters[entity_type]}]"

        entity_map[key] = replacement
        return replacement

    @staticmethod
    def format_log_table(records: list[ReplacementRecord]) -> list[dict]:
        """Format replacement records as a table for UI display."""
        rows = []
        for rec in records:
            rows.append({
                "文本ID": rec.text_id,
                "原文片段": rec.original_span,
                "替换为": rec.replacement,
                "实体类型": rec.entity_type,
                "置信度": f"{rec.confidence:.2f}",
            })
        return rows
