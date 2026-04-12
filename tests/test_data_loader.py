"""Tests for socialscikit.core.data_loader."""

import json
from pathlib import Path

import pandas as pd
import pytest

from socialscikit.core.data_loader import DataLoadError, get_template_path, load_file


@pytest.fixture()
def tmp_dir(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


class TestLoadCSV:
    def test_basic_csv(self, tmp_dir):
        p = tmp_dir / "data.csv"
        p.write_text("id,text,label\n1,hello,pos\n2,world,neg\n", encoding="utf-8")
        df = load_file(p)
        assert list(df.columns) == ["id", "text", "label"]
        assert len(df) == 2

    def test_tab_separated(self, tmp_dir):
        p = tmp_dir / "data.csv"
        p.write_text("id\ttext\tlabel\n1\thello\tpos\n2\tworld\tneg\n", encoding="utf-8")
        df = load_file(p)
        assert "text" in df.columns

    def test_latin1_encoding(self, tmp_dir):
        p = tmp_dir / "data.csv"
        p.write_bytes("id,text\n1,café\n".encode("latin-1"))
        df = load_file(p)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------


class TestLoadExcel:
    def test_basic_xlsx(self, tmp_dir):
        p = tmp_dir / "data.xlsx"
        pd.DataFrame({"text": ["a", "b"]}).to_excel(p, index=False)
        df = load_file(p)
        assert list(df.columns) == ["text"]
        assert len(df) == 2


# ---------------------------------------------------------------------------
# JSON / JSONL
# ---------------------------------------------------------------------------


class TestLoadJSON:
    def test_list_of_objects(self, tmp_dir):
        p = tmp_dir / "data.json"
        p.write_text(json.dumps([{"text": "a"}, {"text": "b"}]), encoding="utf-8")
        df = load_file(p)
        assert len(df) == 2

    def test_data_key(self, tmp_dir):
        p = tmp_dir / "data.json"
        p.write_text(json.dumps({"data": [{"text": "a"}]}), encoding="utf-8")
        df = load_file(p)
        assert len(df) == 1

    def test_invalid_json_structure(self, tmp_dir):
        p = tmp_dir / "data.json"
        p.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        with pytest.raises(DataLoadError, match="list"):
            load_file(p)

    def test_jsonl(self, tmp_dir):
        p = tmp_dir / "data.jsonl"
        lines = [json.dumps({"text": "hello"}), json.dumps({"text": "world"})]
        p.write_text("\n".join(lines), encoding="utf-8")
        df = load_file(p)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------


class TestLoadTXT:
    def test_basic_txt(self, tmp_dir):
        p = tmp_dir / "data.txt"
        p.write_text("line one\nline two\nline three\n", encoding="utf-8")
        df = load_file(p)
        assert list(df.columns) == ["text"]
        assert len(df) == 3

    def test_empty_txt(self, tmp_dir):
        p = tmp_dir / "data.txt"
        p.write_text("", encoding="utf-8")
        with pytest.raises(DataLoadError, match="empty"):
            load_file(p)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_file_not_found(self):
        with pytest.raises(DataLoadError, match="not found"):
            load_file("/nonexistent/file.csv")

    def test_unsupported_extension(self, tmp_dir):
        p = tmp_dir / "data.pdf"
        p.write_text("dummy", encoding="utf-8")
        with pytest.raises(DataLoadError, match="Unsupported"):
            load_file(p)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_quantikit_template_exists(self):
        p = get_template_path("quantikit")
        assert p.exists()

    def test_qualikit_template_exists(self):
        p = get_template_path("qualikit")
        assert p.exists()

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            get_template_path("invalid")
