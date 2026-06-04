"""SQLite-backed local knowledge base for fast keyword matching."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from contextlib import closing
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from .types import LanguageCode

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class KnowledgeRow:
    """A single knowledge-base row retrieved from SQLite."""

    language: str
    keywords: str
    response: str


@dataclass(frozen=True)
class KnowledgeMatch:
    """Best knowledge-base match details used for confidence-gated routing."""

    response: str
    score: float
    matched_keyword: str
    retrieval_mode: str
    source_id: str | None = None
    section: str | None = None
    question: str | None = None
    exact_hit_count: int = 0
    fuzzy_hit_count: int = 0
    matched_keyword_token_count: int = 0
    query_token_count: int = 0
    keyword_coverage: float = 0.0
    query_coverage: float = 0.0
    score_margin: float = 0.0
    whole_phrase_match: bool = False


@dataclass(frozen=True)
class _MatchEvidence:
    """Token evidence behind a keyword match score."""

    score: float
    keyword: str
    exact_hit_count: int
    fuzzy_hit_count: int
    matched_keyword_token_count: int
    query_token_count: int
    keyword_coverage: float
    query_coverage: float
    whole_phrase_match: bool = False


class KnowledgeBase:
    """Manage the local Q&A database and perform keyword lookups."""

    def __init__(
        self,
        db_path: Path,
        retrieval_mode: str = "lexical",
        lexical_top_k: int = 5,
        vector_top_k: int = 5,
        confidence_high: float = 0.8,
        confidence_low: float = 0.55,
    ) -> None:
        """Create a knowledge-base wrapper for the given SQLite path."""

        self.db_path = db_path
        self.retrieval_mode = self._normalize_retrieval_mode(retrieval_mode)
        self.lexical_top_k = max(1, lexical_top_k)
        self.vector_top_k = max(1, vector_top_k)
        self.confidence_high = self._clamp_probability(confidence_high)
        self.confidence_low = self._clamp_probability(confidence_low)
        if self.confidence_low > self.confidence_high:
            self.confidence_low = self.confidence_high

    def ensure_schema(self) -> None:
        """Create the knowledge_base table and index if they do not exist."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    language TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    response TEXT NOT NULL,
                    source_id TEXT,
                    section TEXT,
                    question TEXT
                )
                """
            )
            self._ensure_optional_columns(connection)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_lang_keywords ON knowledge_base(language)"
            )
            connection.commit()

    @staticmethod
    def _ensure_optional_columns(connection: sqlite3.Connection) -> None:
        """Add metadata columns when migrating from older local schemas."""

        cursor = connection.execute("PRAGMA table_info(knowledge_base)")
        existing_columns = {str(row[1]).lower() for row in cursor.fetchall()}
        optional_columns = (
            "source_id TEXT",
            "section TEXT",
            "question TEXT",
        )
        for column_definition in optional_columns:
            column_name = column_definition.split()[0].lower()
            if column_name in existing_columns:
                continue
            connection.execute(f"ALTER TABLE knowledge_base ADD COLUMN {column_definition}")

    def is_empty(self) -> bool:
        """Return whether the knowledge base currently has any rows."""

        self.ensure_schema()
        with closing(sqlite3.connect(self.db_path)) as connection:
            cursor = connection.execute("SELECT COUNT(*) FROM knowledge_base")
            return int(cursor.fetchone()[0]) == 0

    def has_curated_rows(self) -> bool:
        """Return whether curated Q&A seed rows (with source IDs) exist."""

        self.ensure_schema()
        with closing(sqlite3.connect(self.db_path)) as connection:
            cursor = connection.execute(
                "SELECT COUNT(*) FROM knowledge_base WHERE source_id IS NOT NULL AND TRIM(source_id) <> ''"
            )
            return int(cursor.fetchone()[0]) > 0

    def seed_demo_rows(self) -> None:
        """Insert a small demo knowledge set for local-answer testing."""

        rows: Iterable[tuple[str, str, str]] = (
            ("en", "what is your name,who are you", "I am the voice assistant pipeline demo."),
            ("en", "reset password,forgot password,password help", "Use the account settings page to reset your password."),
            ("en", "office hours,opening hours,working hours", "Support is available from 9 AM to 5 PM, Monday through Friday."),
            ("vi", "ban la ai,tên của bạn là gì", "Tôi là bản demo của pipeline trợ lý giọng nói."),
            ("vi", "quen mat khau,đặt lại mật khẩu,reset password", "Hãy vào trang cài đặt tài khoản để đặt lại mật khẩu."),
            ("vi", "gio lam viec,giờ mở cửa,office hours", "Bộ phận hỗ trợ làm việc từ 9 giờ đến 17 giờ, từ thứ Hai đến thứ Sáu."),
        )
        self.ensure_schema()
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executemany(
                "INSERT INTO knowledge_base(language, keywords, response) VALUES (?, ?, ?)",
                list(rows),
            )
            connection.commit()

    def seed_from_qa_json(self, qa_json_path: Path) -> int:
        """Insert curated VN/EN Q&A pairs from a JSON dataset into the local DB."""

        self.ensure_schema()
        if not qa_json_path.exists():
            return 0

        records = json.loads(qa_json_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            return 0

        rows_to_insert: list[tuple[str, str, str, str | None, str | None, str | None]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            source_id = str(record.get("id", "")).strip() or None
            section_vi = str(record.get("section_vi", "")).strip() or None

            question_vi = str(record.get("question_vi", "")).strip()
            answer_vi = str(record.get("answer_vi", "")).strip()
            if question_vi and answer_vi:
                rows_to_insert.append(
                    (
                        "vi",
                        self._build_keywords_from_question(question_vi, "vi"),
                        answer_vi,
                        source_id,
                        section_vi,
                        question_vi,
                    )
                )

            question_en = str(record.get("question_en", "")).strip()
            answer_en = str(record.get("answer_en", "")).strip()
            if question_en and answer_en:
                rows_to_insert.append(
                    (
                        "en",
                        self._build_keywords_from_question(question_en, "en"),
                        answer_en,
                        source_id,
                        section_vi,
                        question_en,
                    )
                )

        if not rows_to_insert:
            return 0

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executemany(
                """
                INSERT INTO knowledge_base(language, keywords, response, source_id, section, question)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            connection.commit()
        return len(rows_to_insert)

    def lookup_question_by_source(self, source_id: str, language: LanguageCode) -> str | None:
        """Return the stored question text for a source ID and language, if available."""

        normalized_source_id = source_id.strip()
        if not normalized_source_id:
            return None

        self.ensure_schema()
        with closing(sqlite3.connect(self.db_path)) as connection:
            cursor = connection.execute(
                """
                SELECT question
                FROM knowledge_base
                WHERE source_id = ? AND language = ? AND question IS NOT NULL AND TRIM(question) <> ''
                LIMIT 1
                """,
                (normalized_source_id, language),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            question = str(row[0]).strip()
            return question or None

    def lookup_response_by_source_details(self, source_id: str, language: LanguageCode) -> KnowledgeMatch | None:
        """Return a curated row by exact source ID and language."""

        normalized_source_id = source_id.strip()
        if not normalized_source_id:
            return None

        self.ensure_schema()
        with closing(sqlite3.connect(self.db_path)) as connection:
            cursor = connection.execute(
                """
                SELECT keywords, response, source_id, section, question
                FROM knowledge_base
                WHERE source_id = ? AND language = ?
                LIMIT 1
                """,
                (normalized_source_id, language),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            keywords, response, stored_source_id, section, question = row
            return KnowledgeMatch(
                response=str(response),
                score=1.0,
                matched_keyword=str(keywords).split(",", 1)[0].strip(),
                retrieval_mode=f"{self.retrieval_mode}_intent",
                source_id=str(stored_source_id).strip() if stored_source_id is not None else None,
                section=str(section).strip() if section is not None else None,
                question=str(question).strip() if question is not None else None,
                exact_hit_count=2,
                fuzzy_hit_count=0,
                matched_keyword_token_count=2,
                query_token_count=2,
                keyword_coverage=1.0,
                query_coverage=1.0,
                score_margin=1.0,
                whole_phrase_match=True,
            )

    def lookup_response(self, query: str, language: LanguageCode) -> str | None:
        """Return the best matching local answer for the supplied query."""

        best_match = self.lookup_response_details(query, language)
        if best_match is None:
            return None
        # Preserve historical behavior of direct DB lookups.
        if best_match.score >= 0.6:
            return best_match.response
        return None

    def lookup_response_details(self, query: str, language: LanguageCode) -> KnowledgeMatch | None:
        """Return best local answer plus confidence metadata for routing decisions."""

        self.ensure_schema()
        query_tokens = self._tokenize(query)
        query_token_list = self._tokenize_list(query)
        normalized_query = self._normalize(query)
        if not normalized_query:
            return None

        row_scores: list[dict[str, str | float | _MatchEvidence]] = []

        with closing(sqlite3.connect(self.db_path)) as connection:
            cursor = connection.execute(
                """
                SELECT language, keywords, response, source_id, section, question
                FROM knowledge_base
                WHERE language = ?
                """,
                (language,),
            )
            for language_value, keywords, response, source_id, section, question in cursor.fetchall():
                lexical_evidence = self._score_match(
                    query_token_list,
                    normalized_query,
                    str(keywords),
                )
                semantic_evidence = self._semantic_score_match(
                    query_token_list,
                    normalized_query,
                    str(keywords),
                )
                lexical_score = lexical_evidence.score
                semantic_score = semantic_evidence.score
                row_scores.append(
                    {
                        "response": str(response),
                        "lexical_score": lexical_score,
                        "semantic_score": semantic_score,
                        "hybrid_score": (0.65 * lexical_score) + (0.35 * semantic_score),
                        "lexical_evidence": lexical_evidence,
                        "semantic_evidence": semantic_evidence,
                        "source_id": str(source_id).strip() if source_id is not None else None,
                        "section": str(section).strip() if section is not None else None,
                        "question": str(question).strip() if question is not None else None,
                    }
                )

        if not row_scores:
            return None

        if self.retrieval_mode == "lexical":
            selected = max(row_scores, key=lambda item: float(item["lexical_score"]))
            ranked_scores = sorted((float(item["lexical_score"]) for item in row_scores), reverse=True)
            return self._build_knowledge_match(selected, "lexical", selected["lexical_evidence"], ranked_scores)

        if self.retrieval_mode == "vector":
            semantic_ranked = sorted(
                row_scores,
                key=lambda item: float(item["semantic_score"]),
                reverse=True,
            )
            selected = semantic_ranked[: self.vector_top_k][0]
            ranked_scores = [float(item["semantic_score"]) for item in semantic_ranked]
            return self._build_knowledge_match(selected, "vector", selected["semantic_evidence"], ranked_scores)

        lexical_ranked = sorted(
            row_scores,
            key=lambda item: float(item["lexical_score"]),
            reverse=True,
        )
        shortlist = lexical_ranked[: self.lexical_top_k] if lexical_ranked else row_scores
        hybrid_ranked = sorted(shortlist, key=lambda item: float(item["hybrid_score"]), reverse=True)
        selected = hybrid_ranked[0]
        lexical_evidence = selected["lexical_evidence"]
        semantic_evidence = selected["semantic_evidence"]
        selected_evidence = (
            lexical_evidence
            if isinstance(lexical_evidence, _MatchEvidence)
            and isinstance(semantic_evidence, _MatchEvidence)
            and lexical_evidence.score >= semantic_evidence.score
            else semantic_evidence
        )
        ranked_scores = [float(item["hybrid_score"]) for item in hybrid_ranked]
        if isinstance(selected_evidence, _MatchEvidence):
            evidence = _MatchEvidence(
                score=float(selected["hybrid_score"]),
                keyword=selected_evidence.keyword,
                exact_hit_count=selected_evidence.exact_hit_count,
                fuzzy_hit_count=selected_evidence.fuzzy_hit_count,
                matched_keyword_token_count=selected_evidence.matched_keyword_token_count,
                query_token_count=selected_evidence.query_token_count,
                keyword_coverage=selected_evidence.keyword_coverage,
                query_coverage=selected_evidence.query_coverage,
                whole_phrase_match=selected_evidence.whole_phrase_match,
            )
        else:
            evidence = self._empty_evidence(query_token_count=len(query_token_list))
        return self._build_knowledge_match(selected, "hybrid", evidence, ranked_scores)

    @staticmethod
    def _build_knowledge_match(
        selected: dict[str, str | float | _MatchEvidence],
        retrieval_mode: str,
        evidence_value: str | float | _MatchEvidence,
        ranked_scores: list[float],
    ) -> KnowledgeMatch:
        """Build a public match object with token-evidence diagnostics."""

        evidence = evidence_value if isinstance(evidence_value, _MatchEvidence) else KnowledgeBase._empty_evidence()
        score_margin = ranked_scores[0] - ranked_scores[1] if len(ranked_scores) > 1 else ranked_scores[0]
        return KnowledgeMatch(
            response=str(selected["response"]),
            score=float(ranked_scores[0]) if ranked_scores else evidence.score,
            matched_keyword=evidence.keyword,
            retrieval_mode=retrieval_mode,
            source_id=selected["source_id"] if isinstance(selected["source_id"], str) else None,
            section=selected["section"] if isinstance(selected["section"], str) else None,
            question=selected["question"] if isinstance(selected["question"], str) else None,
            exact_hit_count=evidence.exact_hit_count,
            fuzzy_hit_count=evidence.fuzzy_hit_count,
            matched_keyword_token_count=evidence.matched_keyword_token_count,
            query_token_count=evidence.query_token_count,
            keyword_coverage=evidence.keyword_coverage,
            query_coverage=evidence.query_coverage,
            score_margin=score_margin,
            whole_phrase_match=evidence.whole_phrase_match,
        )

    @staticmethod
    def _empty_evidence(query_token_count: int = 0) -> _MatchEvidence:
        """Return empty evidence for no usable keyword evidence."""

        return _MatchEvidence(
            score=0.0,
            keyword="",
            exact_hit_count=0,
            fuzzy_hit_count=0,
            matched_keyword_token_count=0,
            query_token_count=query_token_count,
            keyword_coverage=0.0,
            query_coverage=0.0,
        )

    @classmethod
    def _build_keywords_from_question(cls, question: str, language: LanguageCode) -> str:
        """Derive compact keyword phrases from a curated question for robust local matching."""

        normalized_question = cls._normalize(question)
        if not normalized_question:
            return ""

        stopwords_en = {
            "what",
            "is",
            "are",
            "the",
            "a",
            "an",
            "for",
            "of",
            "to",
            "in",
            "at",
            "do",
            "does",
            "can",
            "how",
            "when",
            "where",
            "who",
            "why",
            "about",
            "with",
            "on",
        }
        stopwords_vi = {
            "la",
            "gi",
            "co",
            "khong",
            "bao",
            "nhieu",
            "nao",
            "o",
            "ve",
            "cho",
            "toi",
            "ban",
            "cua",
            "duoc",
            "khong",
            "hay",
            "neu",
            "thi",
        }
        stopwords = stopwords_vi if language == "vi" else stopwords_en

        all_tokens = normalized_question.split()
        content_tokens = [token for token in all_tokens if token not in stopwords]
        if not content_tokens:
            content_tokens = all_tokens

        phrases: list[str] = [normalized_question]
        for ngram_size in (2, 3):
            for start in range(0, len(content_tokens) - ngram_size + 1):
                phrase = " ".join(content_tokens[start : start + ngram_size])
                if phrase:
                    phrases.append(phrase)

        # Keep the compact set deterministic to avoid excessively broad matching.
        deduped = list(dict.fromkeys(phrase for phrase in phrases if phrase))
        return ",".join(deduped[:10])

    def _score_match(self, query_tokens: list[str], normalized_query: str, keywords: str) -> _MatchEvidence:
        """Score a row by exact/fuzzy token evidence against keyword phrases."""

        best_evidence = self._empty_evidence(query_token_count=len(query_tokens))
        for raw_keyword in keywords.split(","):
            normalized_keyword = self._normalize(raw_keyword)
            if not normalized_keyword:
                continue
            keyword_tokens = normalized_keyword.split()
            if not keyword_tokens:
                continue
            evidence = self._keyword_evidence(query_tokens, keyword_tokens, normalized_keyword)
            if evidence.score > best_evidence.score:
                best_evidence = evidence
        return best_evidence

    def _semantic_score_match(self, query_tokens: list[str], normalized_query: str, keywords: str) -> _MatchEvidence:
        """Score a row with token evidence plus a small phrase-similarity tiebreaker."""

        best_evidence = self._empty_evidence(query_token_count=len(query_tokens))
        for raw_keyword in keywords.split(","):
            normalized_keyword = self._normalize(raw_keyword)
            if not normalized_keyword:
                continue
            keyword_tokens = normalized_keyword.split()
            if not keyword_tokens:
                continue
            evidence = self._keyword_evidence(query_tokens, keyword_tokens, normalized_keyword)
            phrase_similarity = SequenceMatcher(None, normalized_query, normalized_keyword).ratio()
            score = min(1.0, evidence.score + (0.05 * phrase_similarity))
            evidence = _MatchEvidence(
                score=score,
                keyword=evidence.keyword,
                exact_hit_count=evidence.exact_hit_count,
                fuzzy_hit_count=evidence.fuzzy_hit_count,
                matched_keyword_token_count=evidence.matched_keyword_token_count,
                query_token_count=evidence.query_token_count,
                keyword_coverage=evidence.keyword_coverage,
                query_coverage=evidence.query_coverage,
                whole_phrase_match=evidence.whole_phrase_match,
            )
            if evidence.score > best_evidence.score:
                best_evidence = evidence
        return best_evidence

    @classmethod
    def _keyword_evidence(
        cls,
        query_tokens: list[str],
        keyword_tokens: list[str],
        normalized_keyword: str,
    ) -> _MatchEvidence:
        """Return token-evidence score for one normalized keyword phrase."""

        query_content_tokens = cls._content_tokens(query_tokens)
        keyword_content_tokens = cls._content_tokens(keyword_tokens)
        evidence_keyword_tokens = keyword_content_tokens or keyword_tokens
        evidence_query_tokens = query_content_tokens or query_tokens
        if not evidence_keyword_tokens or not evidence_query_tokens:
            return cls._empty_evidence(query_token_count=len(evidence_query_tokens))

        exact_hits = sum(1 for token in evidence_keyword_tokens if token in evidence_query_tokens)
        fuzzy_hits = cls._fuzzy_token_hit_count(evidence_query_tokens, evidence_keyword_tokens)
        keyword_count = len(evidence_keyword_tokens)
        query_count = len(evidence_query_tokens)
        keyword_coverage = min(1.0, (exact_hits + fuzzy_hits) / float(keyword_count))
        query_coverage = min(1.0, (exact_hits + fuzzy_hits) / float(query_count))
        whole_phrase_match = cls._contains_token_sequence(query_tokens, keyword_tokens)
        if whole_phrase_match:
            score = 1.0
        else:
            exact_component = exact_hits / float(keyword_count)
            fuzzy_component = fuzzy_hits / float(keyword_count)
            score = min(1.0, exact_component + (0.35 * fuzzy_component))
            if keyword_count == 1:
                score = min(score, 0.50)
        return _MatchEvidence(
            score=score,
            keyword=normalized_keyword,
            exact_hit_count=exact_hits,
            fuzzy_hit_count=fuzzy_hits,
            matched_keyword_token_count=keyword_count,
            query_token_count=query_count,
            keyword_coverage=keyword_coverage,
            query_coverage=query_coverage,
            whole_phrase_match=whole_phrase_match,
        )

    @staticmethod
    def _fuzzy_token_hit_count(query_tokens: list[str], keyword_tokens: list[str]) -> int:
        """Return fuzzy-only token hits, excluding exact hits."""

        if not query_tokens or not keyword_tokens:
            return 0
        matched = 0
        for keyword_token in keyword_tokens:
            if keyword_token in query_tokens:
                continue
            if len(keyword_token) < 4:
                continue
            if any(
                len(query_token) >= 4 and SequenceMatcher(None, keyword_token, query_token).ratio() >= 0.92
                for query_token in query_tokens
            ):
                matched += 1
        return matched

    @staticmethod
    def _contains_token_sequence(query_tokens: list[str], keyword_tokens: list[str]) -> bool:
        """Return whether keyword tokens appear as a whole-token contiguous sequence."""

        if not query_tokens or not keyword_tokens or len(keyword_tokens) > len(query_tokens):
            return False
        window_size = len(keyword_tokens)
        return any(
            query_tokens[index : index + window_size] == keyword_tokens
            for index in range(0, len(query_tokens) - window_size + 1)
        )

    @classmethod
    def _content_tokens(cls, tokens: list[str]) -> list[str]:
        """Remove generic prompt-openers and function words from scoring evidence."""

        ignored_tokens = {
            "a",
            "an",
            "and",
            "are",
            "about",
            "ban",
            "biet",
            "can",
            "cho",
            "could",
            "cua",
            "do",
            "does",
            "duoc",
            "for",
            "gi",
            "how",
            "i",
            "in",
            "is",
            "it",
            "la",
            "me",
            "on",
            "please",
            "tell",
            "the",
            "thi",
            "to",
            "toi",
            "ve",
            "what",
            "with",
            "you",
        }
        return [token for token in tokens if token not in ignored_tokens]

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text into lowercase ASCII-ish tokens for comparison."""

        lowered = text.lower().replace("đ", "d")
        ascii_like = unicodedata.normalize("NFD", lowered).encode("ascii", "ignore").decode("ascii")
        return " ".join(_TOKEN_RE.findall(ascii_like))

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        """Split text into normalized tokens for keyword matching."""

        return set(cls._normalize(text).split())

    @classmethod
    def _tokenize_list(cls, text: str) -> list[str]:
        """Split text into ordered normalized tokens for phrase matching."""

        return cls._normalize(text).split()

    @staticmethod
    def _normalize_retrieval_mode(value: str) -> str:
        """Return safe retrieval mode value from configuration."""

        normalized = value.strip().lower()
        if normalized in {"lexical", "vector", "hybrid"}:
            return normalized
        return "lexical"

    @staticmethod
    def _clamp_probability(value: float) -> float:
        """Clamp probability-like values into [0.0, 1.0]."""

        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
