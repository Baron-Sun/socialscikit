<p align="center">
  <h1 align="center">SocialSciKit</h1>
  <p align="center">
    <strong>A zero-code text analysis toolkit for social science researchers</strong>
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
  <strong>English</strong> | <a href="README_zh.md">中文文档</a>
</p>

---

## What is SocialSciKit?

SocialSciKit is an open-source Python toolkit that enables social science researchers to perform text analysis **without writing a single line of code**. It provides a Gradio-based web interface with full bilingual support (English / Chinese).

Three core modules:

- **QuantiKit** — End-to-end text classification pipeline (method recommendation &rarr; annotation &rarr; prompt/fine-tuning classification &rarr; evaluation &rarr; export)
- **QualiKit** — End-to-end qualitative coding pipeline (upload &rarr; de-identification &rarr; research framework &rarr; LLM coding &rarr; human review &rarr; export)
- **Toolbox** — Standalone research methods tools: Inter-Coder Reliability (ICR) calculator, Multi-LLM Consensus Coding, and Methods Section Generator

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [QuantiKit: Text Classification](#quantikit-text-classification)
- [QualiKit: Qualitative Coding](#qualikit-qualitative-coding)
- [Toolbox: Research Methods Tools](#toolbox-research-methods-tools)
- [Supported LLM Backends](#supported-llm-backends)
- [Example Datasets](#example-datasets)
- [Project Structure](#project-structure)
- [Key References](#key-references)
- [Citation](#citation)
- [Development](#development)
- [License & Disclaimer](#license--disclaimer)
- [Author](#author)

---

## Installation

### Requirements

- Python 3.9 or higher
- pip (Python package manager)

### Option A: Install from PyPI

```bash
pip install socialscikit
```

### Option B: Install from source

```bash
git clone https://github.com/Baron-Sun/socialscikit.git
cd socialscikit
pip install -e .
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `gradio` | &ge; 4.0 | Web UI framework |
| `pandas` | &ge; 2.0 | Data manipulation |
| `openpyxl` | any | Excel read/write |
| `spacy` | &ge; 3.7 | NLP pipeline (tokenization, NER) |
| `transformers` | &ge; 4.40 | Fine-tuning (RoBERTa / XLM-R) |
| `datasets` | any | HuggingFace dataset handling |
| `openai` | &ge; 1.0 | OpenAI API client |
| `anthropic` | &ge; 0.25 | Anthropic API client |
| `scikit-learn` | any | Evaluation metrics |
| `scipy` | any | Statistical computation |
| `bertopic` | any | Topic modeling |
| `presidio-analyzer` | any | PII detection engine |
| `presidio-anonymizer` | any | PII anonymization |
| `langdetect` | any | Language detection |
| `tiktoken` | any | Token counting |
| `httpx` | any | Ollama HTTP client |
| `rich` | any | CLI formatting |

### Optional: spaCy language models

For best de-identification performance, download at least one spaCy model:

```bash
# English
python -m spacy download en_core_web_sm

# Chinese
python -m spacy download zh_core_web_sm
```

---

## Quick Start

### Launch the unified app (recommended)

```bash
socialscikit launch
# or simply:
socialscikit
# Opens at http://127.0.0.1:7860
```

### Launch individual modules

```bash
# QuantiKit only
socialscikit quantikit --port 7860

# QualiKit only
socialscikit qualikit --port 7861
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--port` | Server port number | 7860 / 7861 |
| `--share` | Create a public Gradio link | `False` |

### First-time language switch

The default UI language is **English**. Use the **Language** toggle at the top of the page to switch to **Chinese**. All labels, buttons, and instructions update in real time.

---

## QuantiKit: Text Classification

QuantiKit guides you through the full text classification workflow in 6 steps.

### Step 1 &middot; Data Upload

- **Supported formats**: CSV, Excel (.xlsx/.xls), JSON, JSONL
- Upload your data file, then map the `text` and `label` columns
- Automatic data validation: detects missing values, empty strings, encoding issues
- **One-click fix**: auto-repair common data quality issues
- Diagnostic report: label distribution, text length statistics, duplicate detection

### Step 2 &middot; Recommendation

- **Method recommender**: analyzes your data characteristics (size, class count, imbalance ratio, text length) and recommends the optimal classification approach — zero-shot, few-shot, or fine-tuning — with literature citations
- **Budget recommender**: estimates "how many labels do you need?" using power-law learning curve fitting, with 80% confidence intervals and marginal return curves
  - *Cold-start mode*: priors from CSS benchmark datasets (HatEval, SemEval, MFTC)
  - *Empirical mode*: fits `f1 = a * n^b + c` on your labeled subset

### Step 3 &middot; Annotation

- Built-in annotation UI — no need for external tools
- Label each text sample, with **skip**, **undo**, **flag for review** support
- Export annotated data as CSV, merge with original dataset
- Progress tracker shows completion percentage

### Step 4 &middot; Classification

Three sub-approaches available in parallel tabs:

| Sub-tab | Method | When to use |
|---------|--------|-------------|
| **Prompt Classification** | Zero/few-shot via LLM API | Small datasets (&lt; 200 labeled) |
| **Fine-tuning** | Local transformer fine-tuning | Medium datasets (200+), no API cost |
| **API Fine-tuning** | OpenAI fine-tuning API | Large datasets, best performance |

**Prompt Classification** features:
- Prompt Designer: task description + class definitions + positive/negative examples &rarr; auto-generates a structured prompt
- Prompt Optimizer: generates 3 variants using APE (Automatic Prompt Engineering), evaluates each on a test split
- One-click batch classification on the full dataset

### Step 5 &middot; Evaluation

- Metrics: Accuracy, Macro-F1, Cohen's Kappa, per-class Precision / Recall / F1
- Confusion matrix visualization
- Detailed classification report

### Step 6 &middot; Export

- Download classification results as CSV (original text + predicted labels + confidence)

---

## QualiKit: Qualitative Coding

QualiKit supports the full qualitative coding workflow for interview transcripts, focus group data, and open-ended survey responses.

### Step 1 &middot; Upload & Segment

- **Supported formats**: plain text (.txt)
- Automatic speaker detection and segmentation (by paragraph or by speaker turn)
- Configurable context window (number of surrounding sentences to include)
- Preview segmented results in a table before proceeding

### Step 2 &middot; De-identification

- Automatic PII detection: person names, email addresses, phone numbers, Chinese ID card numbers
- Chinese-aware NER: detects Chinese names with title/honorific patterns
- English NER via spaCy and Presidio
- Replacement strategies: pseudonym, redact (`[REDACTED]`), or tag-based (`[PERSON_1]`)
- **Per-item review**: accept, reject, or edit each detected PII replacement individually
- **Bulk actions**: accept all, accept high-confidence only (&ge; 0.90), or apply all accepted to the text

### Step 3 &middot; Research Framework

- Define your Research Questions (RQs) and Sub-themes using an interactive editable table
- Add/remove rows dynamically
- **LLM-powered sub-theme suggestion**: connect to an LLM backend, and it analyzes your transcript to suggest relevant sub-themes per RQ
- Confirm framework before proceeding to coding

### Step 4 &middot; LLM Coding

- Batch coding: LLM reads each segment and assigns RQ + sub-theme labels with confidence scores
- Supports OpenAI, Anthropic, and Ollama backends
- Results displayed with segment text, assigned codes, and confidence levels

### Step 5 &middot; Review

- Review coding results in a table sorted by confidence
- Per-item actions: accept, reject, or edit (reassign RQ/sub-theme)
- Bulk accept by confidence threshold
- **Manual coding**: select a segment, preview its content, and manually assign RQ + sub-theme labels
- Cascading dropdown: sub-theme choices automatically filter based on selected RQ

### Step 6 &middot; Export

- Export reviewed coding results as structured Excel file

---

## Toolbox: Research Methods Tools

The Toolbox provides standalone research utilities that work independently or in combination with QuantiKit / QualiKit.

### ICR Calculator

Compute inter-coder reliability for 2 or more coders with automatic metric selection:

| Scenario | Metric |
|----------|--------|
| 2 coders, single-label | Cohen's Kappa + Krippendorff's Alpha + per-category agreement |
| 3+ coders, single-label | Krippendorff's Alpha + pairwise Cohen's Kappa |
| 2 coders, multi-label | Jaccard index (pairwise) |
| 3+ coders, multi-label | Average pairwise Jaccard |

- Upload a CSV with coder columns, select which columns to compare
- Interpretation follows the Landis & Koch (1977) scale

### Consensus Coding

Multi-LLM majority-vote coding for qualitative data:

- Configure 2&ndash;5 LLM backends (OpenAI, Anthropic, Ollama) with independent models
- Each LLM codes every text segment; final label is determined by majority vote
- Agreement statistics across LLMs are reported automatically

### Methods Section Generator

Auto-generate a methods section paragraph (English + Chinese) for your paper:

- **From pipeline log**: QuantiKit and QualiKit can export a pipeline log (JSON) capturing all metadata (sample size, model, metrics, themes, etc.). Import the log and generate a ready-to-use methods paragraph.
- **Manual input**: Fill in metadata fields manually if you prefer not to use the pipeline log.

---

## Supported LLM Backends

| Backend | Example Models | Use Case |
|---------|---------------|----------|
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano` | Classification, coding, prompt optimization |
| **Anthropic** | `claude-sonnet-4-20250514`, `claude-haiku-4-5-20251001` | Classification, coding, prompt optimization |
| **Ollama** | `llama3`, `mistral`, `qwen2.5` | Local inference, no API key needed |

To use Ollama, install it from [ollama.com](https://ollama.com) and pull a model:

```bash
ollama pull llama3
```

---

## Example Datasets

The `examples/` directory contains ready-to-use sample data:

| File | Module | Description |
|------|--------|-------------|
| `sentiment_example.csv` | QuantiKit | 50 Chinese product/service reviews with 3 sentiment labels |
| `policy_example.csv` | QuantiKit | 40 Chinese policy text excerpts with 8 policy-instrument labels |
| `interview_example.txt` | QualiKit | Single-person community healthcare interview transcript |
| `interview_focus_group.txt` | QualiKit | 4-person focus group on elderly digital service experiences |
| `icr_example.csv` | Toolbox | 20 policy texts coded by 3 coders (A/B/C) for ICR calculation |
| `consensus_example.csv` | Toolbox | 15 interview segments for multi-LLM consensus coding |
| `methods_log_quantikit.json` | Toolbox | Sample QuantiKit pipeline log for methods generation |
| `methods_log_qualikit.json` | Toolbox | Sample QualiKit pipeline log for methods generation |

### Cookbook: Sentiment Classification (QuantiKit)

1. Launch: `socialscikit launch` &rarr; click **QuantiKit** tab
2. Upload `examples/sentiment_example.csv`
3. Map columns: text &rarr; `text`, label &rarr; `label`
4. Go to **Step 2** &rarr; click **Recommend** to see method suggestion
5. Go to **Step 4** &rarr; select LLM backend &rarr; enter labels
6. Click **Generate Prompt** &rarr; **Run Classification**
7. Go to **Step 5** &rarr; evaluate against gold labels
8. Go to **Step 6** &rarr; export results

### Cookbook: Focus Group Coding (QualiKit)

1. Launch: `socialscikit launch` &rarr; click **QualiKit** tab
2. Upload `examples/interview_focus_group.txt`
3. **Step 1**: select "Speaker turn" segmentation &rarr; click **Segment**
4. **Step 2**: run de-identification &rarr; review and accept/reject each PII replacement
5. **Step 3**: define RQs and sub-themes &rarr; optionally use LLM to suggest sub-themes
6. **Step 4**: select LLM backend &rarr; run batch coding
7. **Step 5**: review results, bulk accept high-confidence codes, manually fix low-confidence ones
8. **Step 6**: export to Excel

---

## Project Structure

```
socialscikit/
├── core/                         # Shared infrastructure
│   ├── data_loader.py            # Multi-format data reader (CSV/Excel/JSON/txt)
│   ├── data_validator.py         # Schema validation + auto-fix
│   ├── data_diagnostics.py       # Data quality diagnostics report
│   ├── llm_client.py             # Unified LLM client (OpenAI/Anthropic/Ollama)
│   ├── icr.py                    # Inter-coder reliability (Kappa/Alpha/Jaccard)
│   ├── methods_writer.py         # Methods section generator (EN/ZH templates)
│   └── templates/                # Template files for download
│
├── quantikit/                    # Text classification module
│   ├── feature_extractor.py      # Dataset feature extraction
│   ├── method_recommender.py     # Rule-based method recommendation (with citations)
│   ├── budget_recommender.py     # Annotation budget estimation
│   ├── prompt_optimizer.py       # APE-based prompt generation & optimization
│   ├── prompt_classifier.py      # Zero/few-shot LLM classification
│   ├── annotator.py              # Built-in annotation interface
│   ├── classifier.py             # Transformer fine-tuning pipeline
│   ├── api_finetuner.py          # OpenAI fine-tuning API wrapper
│   └── evaluator.py              # Accuracy / F1 / Kappa / confusion matrix
│
├── qualikit/                     # Qualitative coding module
│   ├── segmenter.py              # Text segmentation (paragraph / speaker turn)
│   ├── segment_extractor.py      # Segment-level extraction
│   ├── deidentifier.py           # PII detection (Chinese + English)
│   ├── deident_reviewer.py       # De-identification interactive review
│   ├── theme_definer.py          # Theme definition + LLM suggestion
│   ├── theme_reviewer.py         # Theme review & overlap detection
│   ├── coder.py                  # LLM batch coding
│   ├── confidence_ranker.py      # Confidence scoring & ranking
│   ├── coding_reviewer.py        # Human-in-the-loop coding review
│   ├── extraction_reviewer.py    # Extraction result review
│   ├── consensus.py              # Multi-LLM consensus coding (majority vote)
│   └── exporter.py               # Excel / Markdown export
│
├── ui/                           # Gradio web interface
│   ├── main_app.py               # Unified app (Home + QuantiKit + QualiKit + Toolbox)
│   ├── quantikit_app.py          # QuantiKit UI callbacks
│   ├── qualikit_app.py           # QualiKit UI callbacks
│   ├── toolbox_app.py            # Toolbox UI callbacks (ICR/Consensus/Methods)
│   └── i18n.py                   # Internationalization (EN / ZH)
│
├── cli.py                        # Command-line entry point
│
examples/                         # Sample datasets
tests/                            # Test suite (562 tests)
pyproject.toml                    # Package metadata & dependencies
CITATION.cff                      # Citation metadata
```

---

## Key References

The method recommendation engine and workflow design are grounded in the following computational social science literature:

- Sun, B., Chang, C., Ang, Y. Y., Mu, R., Xu, Y. & Zhang, Z. (2026). Creation of the Chinese Adaptive Policy Communication Corpus. *ACL 2026*.
- Carlson, K. et al. (2026). The use of LLMs to annotate data in management research. *Strategic Management Journal*.
- Chae, Y. & Davidson, T. (2025). Large Language Models for text classification. *Sociological Methods & Research*.
- Do, S., Ollion, E. & Shen, R. (2024). The augmented social scientist. *Sociological Methods & Research*, 53(3).
- Dunivin, Z. O. (2024). Scalable qualitative coding with LLMs. *arXiv:2401.15170*.
- Montgomery, J. M. et al. (2024). Improving probabilistic models in text classification via active learning. *APSR*.
- Than, N. et al. (2025). Updating 'The Future of Coding'. *Sociological Methods & Research*.
- Ziems, C. et al. (2024). Can LLMs transform computational social science? *Computational Linguistics*, 50(1).
- Zhou, Y. et al. (2022). Large Language Models are human-level prompt engineers. *ICLR 2023*.

---

## Citation

If you use SocialSciKit in your research, please cite:

```bibtex
@inproceedings{sun2026creation,
  title     = {Creation of the {Chinese} Adaptive Policy Communication Corpus},
  author    = {Sun, Bolun and Chang, Charles and Ang, Yuen Yuen and Mu, Ruotong and Xu, Yuchen and Zhang, Zhengxin},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
  year      = {2026}
}
```

---

## Development

```bash
# Clone the repository
git clone https://github.com/Baron-Sun/socialscikit.git
cd socialscikit

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the full test suite
pytest tests/ -v

# Code style check
ruff check .
```

### Running the app in development mode

```bash
python -c "from socialscikit.ui.main_app import create_app; create_app().launch()"
```

---

## License & Disclaimer

**License**: MIT

**Disclaimer**:

- **De-identification module**: Automatic PII detection is a preliminary processing tool. Manual review is mandatory before IRB submission. This tool does not guarantee complete removal of all identifying information.
- **LLM classification / coding**: Results should be treated as research assistance. Critical research conclusions require human validation.
- **Budget recommendation**: Based on statistical estimation. Actual requirements may vary depending on task complexity and data characteristics.

---

## Author

**Bolun Sun** (孙博伦)

Ph.D. Student, [Kellogg School of Management](https://www.kellogg.northwestern.edu/), Northwestern University

Research interests: Computational Social Science, NLP, Human-Centered AI

Email: bolun.sun@kellogg.northwestern.edu | Web: [baron-sun.github.io](https://baron-sun.github.io/)

---

## Contributing

This project is actively maintained and updated. Contributions, suggestions, and feedback are very welcome! Feel free to [open an issue](https://github.com/Baron-Sun/socialscikit/issues) or submit a pull request.
