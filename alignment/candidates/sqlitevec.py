"""
Sqlite fulltext and vector search with Spacy lemmatizer

# taken from https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/vectorstores/sqlitevec.py
"""

import json
import logging
import struct
import warnings
import re
import unicodedata
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)
from tqdm.auto import tqdm
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from sqlitefts import fts5
import spacy
import sqlite3

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)




class SpacyENTokenizer(fts5.FTS5Tokenizer):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize(self, text, flags=None):
        doc = self.nlp(text)
        for token in doc:
            t = token.lemma_ # lemmatize
            p = len(text[:token.idx].encode('utf-8')) #byte length
            l = len(token.text.encode('utf-8')) #byte length
            yield t, p, p + l


class SpacyDETokenizer(fts5.FTS5Tokenizer):
    def __init__(self):
        self.nlp = spacy.load("de_core_news_sm")

    def tokenize(self, text, flags=None):
        doc = self.nlp(text)
        for token in doc:
            t = token.lemma_ if token.lemma_ != '--' else token.text
            p = len(text[:token.idx].encode('utf-8')) #byte length
            l = len(token.text.encode('utf-8')) #byte length
            yield t, p, p + l


def serialize_f32(vector: List[float]) -> bytes:
    """Serializes a list of floats into a compact "raw bytes" format

    Source: https://github.com/asg017/sqlite-vec/blob/21c5a14fc71c83f135f5b00c84115139fd12c492/examples/simple-python/demo.py#L8-L10
    """
    return struct.pack("%sf" % len(vector), *vector)


def _normalize(text: str) -> str:
    return unicodedata.normalize('NFKD', text).strip()


class SQLiteVec(VectorStore):
    """SQLite with Vec extension as a vector database.

    To use, you should have the ``sqlite-vec`` python package installed.
    Example:
        .. code-block:: python
            from langchain_community.vectorstores import SQLiteVec
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            ...
    """

    def __init__(
        self,
        table: str,
        connection: Optional[sqlite3.Connection] = None,
        #embedding: Embeddings,
        db_file: str = "vec.db",
        lang: str = "en",
        distance_metric: str = "l2",
    ):
        """Initialize with sqlite client with vss extension."""
        try:
            import sqlite_vec  # noqa  # pylint: disable=unused-import
        except ImportError:
            raise ImportError(
                "Could not import sqlite-vec python package. "
                "Please install it with `pip install sqlite-vec`."
            )

        if not connection:
            connection = self.create_connection(db_file, lang)

        #if not isinstance(embedding, Embeddings):
        #    warnings.warn("embeddings input must be Embeddings object.")

        self._connection = connection
        self._table = table
        #self._embedding = embedding
        self._distance_metric = distance_metric
        assert distance_metric in ['l1', 'l2', 'cosine'], f'{distance_metric} not supported.'

        self.create_table_if_not_exists()

    def create_table_if_not_exists(self) -> None:
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table}
            (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                title_lower TEXT,
                text TEXT NOT NULL,
                metadata BLOB,
                text_embedding BLOB
            )
            ;
            """
        )
        self._connection.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self._table}_vec USING vec0(
                rowid INTEGER PRIMARY KEY,
                text_embedding float[{self.get_dimensionality()}]
                distance_metric={self._distance_metric}
            )
            ;
            """
        )
        self._connection.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self._table}_fts USING fts5(
                title,
                title_lower,
                text,
                content={self._table},
                tokenize = 'spacy_tokenizer'
            )
            ;
            """
        )
        self._connection.execute(
            f"""
                CREATE TRIGGER IF NOT EXISTS {self._table}_embed_text 
                AFTER INSERT ON {self._table}
                BEGIN
                    INSERT INTO {self._table}_vec(rowid, text_embedding)
                    VALUES (new.rowid, new.text_embedding) 
                    ;
                    INSERT INTO {self._table}_fts(rowid, title, title_lower, text)
                    VALUES (new.rowid, new.title, new.title_lower, new.text) 
                    ;
                END;
            """
        )
        self._connection.commit()

    def add_texts(
        self,
        titles: Iterable[str],
        texts: Iterable[str],
        embeds: List[List[float]] = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the vectorstore index.
        Args:
            titles: Iterable of strings to add to the vectorstore.
            texts: Iterable of strings to add to the vectorstore.
            embeds: List of float numbers. vector representation of texts.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        max_id = self._connection.execute(
            f"SELECT max(rowid) as rowid FROM {self._table}"
        ).fetchone()["rowid"]
        if max_id is None:  # no text added yet
            max_id = 0

        #embeds = self._embedding.embed_documents(list(texts))
        #if not metadatas:
        #    metadatas = [{} for _ in texts]
        
        assert len(titles) == len(texts) == len(embeds) == len(metadatas)
        data_input = [
            (_normalize(title), _normalize(title).lower(), _normalize(text), json.dumps(metadata), serialize_f32(embed))
            for title, text, metadata, embed in zip(titles, texts, metadatas, embeds)
        ]
        self._connection.executemany(
            f"INSERT INTO {self._table}(title, title_lower, text, metadata, text_embedding) VALUES (?,?,?,?,?)",
            data_input,
        )
        self._connection.commit()
        # pulling every ids we just inserted
        results = self._connection.execute(
            f"SELECT rowid FROM {self._table} WHERE rowid > {max_id}"
        )
        return [row["rowid"] for row in results]

    #def similarity_search_with_score_by_vector(
    #    self, embedding: List[float], k: int = 4, **kwargs: Any
    #) -> List[Tuple[Document, float]]:
    #    sql_query = f"""
    #        SELECT 
    #            text,
    #            metadata,
    #            distance
    #        FROM {self._table} AS e
    #        INNER JOIN {self._table}_vec AS v on v.rowid = e.rowid  
    #        WHERE
    #            v.text_embedding MATCH ?
    #            AND k = ?
    #        ORDER BY distance
    #    """
    #    cursor = self._connection.cursor()
    #    cursor.execute(
    #        sql_query,
    #        [serialize_f32(embedding), k],
    #    )
    #    results = cursor.fetchall()
    #
    #    documents = []
    #    for row in results:
    #        metadata = json.loads(row["metadata"]) or {}
    #        doc = Document(page_content=row["text"], metadata=metadata)
    #        documents.append((doc, row["distance"]))
    #
    #    return documents

    def _get_metadata_attributes(self, **kwargs) -> Tuple[List]:
        attributes = []
        
        for k, v in kwargs.items():
            if isinstance(v, list):
                ids = ', '.join([f"'{i}'" if isinstance(i, str) else str(i) for i in v])
                query = f"json_extract(e.metadata, '$.{k}') IN ({ids})"
            elif isinstance(v, str):
                query = f"json_extract(e.metadata, '$.{k}') IS '{v}'"
            elif isinstance(v, int) or isinstance(v, float):
                query = f"json_extract(e.metadata, '$.{k}') = {v}"
            else:
                raise ValueError
            #if k in ['instance_id', 'paragraph_id']:
            #    assert isinstance(v, list)
            #    ids = ', '.join([f"'{i}'" for i in v])
            #    query = f"json_extract(e.metadata, '$.{k}') IN ({ids})"
            #elif k in ['language', 'audience']:
            #    assert isinstance(v, str)
            #    query = f"json_extract(e.metadata, '$.{k}') IS '{v}'"
            #elif k in ["type", "title"]:
            #    assert isinstance(v, str)
            #    query = f"json_extract(e.metadata, '$.{k}') LIKE '%{v}%'"
            attributes.append(query)

        metadata_attributes = ""
        if len(attributes) > 0:
            metadata_attributes = " AND ".join(attributes)
        return metadata_attributes

    def fulltext_search_with_highlight(
        self, query: str, limit: int = None, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        # subquery: filter by metadata
        subquery = ""
        if len(kwargs) > 0:
            metadata_attributes = self._get_metadata_attributes(**kwargs)
            subquery = f"""
                e.rowid IN (
                    SELECT e.rowid 
                    FROM {self._table} AS e 
                    WHERE {metadata_attributes}
                )
                AND """

        limit_query = f"LIMIT {limit}\n" if limit else ""

        # main query: fulltext search
        sql_query = f"""
            SELECT
                highlight({self._table}_fts, 0, '<b>', '</b>') AS highlighted_title,
                highlight({self._table}_fts, 2, '<b>', '</b>') AS highlighted_text,
                e.metadata,
                rank
            FROM {self._table} AS e
            INNER JOIN {self._table}_fts AS t on t.rowid = e.rowid
            WHERE {subquery}{self._table}_fts MATCH ?
            ORDER BY rank
            {limit_query};
        """
        #logger.info(sql_query)
        cursor = self._connection.cursor()
        cursor.execute(sql_query, [query])
        results = cursor.fetchall()

        found = []
        for row in results:
            metadata = json.loads(row["metadata"]) or {}
            metadata['title'] = row["highlighted_title"]
            metadata['text'] = row["highlighted_text"]
            metadata['text_offset'] = self.get_offset_positions(row["highlighted_text"])
            metadata['rank'] = row["rank"]
            found.append(metadata)
        return found

    @staticmethod
    def get_offset_positions(highlighted_text: str) -> List[Tuple[int]]:
        import re
        positions = []
        original_text = highlighted_text.replace('<b>', '').replace('</b>', '')
        matches = re.finditer(r'<b>[^<>]+<\/b>', highlighted_text)
        if matches:
            offset = 0
            for i, m in enumerate(matches, start=1):
                length = len(m.group()) - 7
                start = m.start() - offset
                end = start+length
                offset += 7
                assert m.group()[3:-4] == original_text[start:end]
                #print(m.group(), m.start(), m.end(), original_text[start:end])
                positions.append((start, end))
        return positions

    def metadata_search(self, **kwargs: Any) -> List[Dict[str, Any]]:
        metadata_attributes = self._get_metadata_attributes(**kwargs)

        sql_query = f"""
            SELECT
                e.title,
                e.text,
                e.metadata,
                v.text_embedding
            FROM {self._table} AS e
            INNER JOIN {self._table}_vec AS v on v.rowid = e.rowid
            WHERE {metadata_attributes}
            ;
        """
        #logger.info(sql_query)
        cursor = self._connection.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()

        found = []
        for row in results:
            metadata = json.loads(row["metadata"]) or {}
            metadata['title'] = row["title"]
            metadata['text'] = row["text"]
            emb = struct.unpack("%sf" % self.get_dimensionality(), row['text_embedding'])
            metadata['embedding'] = list(emb)
            found.append(metadata)
        return found

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Dict[str, Any]]:

        # subquery: filter by metadata
        subquery = ""
        if len(kwargs) > 0:
            metadata_attributes = self._get_metadata_attributes(**kwargs)
            subquery = f"""
                e.rowid IN (
                    SELECT e.rowid 
                    FROM {self._table} AS e 
                    WHERE {metadata_attributes}
                )
                AND """

        # main query: search by similarity
        sql_query = f"""
            SELECT
                title,
                text,
                metadata,
                distance
            FROM {self._table} AS e
            INNER JOIN {self._table}_vec AS v on v.rowid = e.rowid
            WHERE
                {subquery}v.text_embedding MATCH ?
                AND k = ?
            ORDER BY distance
            ;
        """
        #logger.info(sql_query)
        cursor = self._connection.cursor()
        cursor.execute(sql_query, [serialize_f32(embedding), k])
        results = cursor.fetchall()

        found = []
        for row in results:
            metadata = json.loads(row["metadata"]) or {}
            metadata['title'] = row["title"]
            metadata['text'] = row["text"]
            metadata['distance'] = row["distance"]
            found.append(metadata)
        return found

    # Don't use this!
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        pass

    #def similarity_search(
    #    self, query: str, k: int = 4, **kwargs: Any
    #) -> List[Document]:
    #    """Return docs most similar to query."""
    #    embedding = self._embedding.embed_query(query)
    #    documents = self.similarity_search_with_score_by_vector(
    #        embedding=embedding, k=k
    #    )
    #    return [doc for doc, _ in documents]

    #def similarity_search_with_score(
    #    self, query: str, k: int = 4, **kwargs: Any
    #) -> List[Tuple[Document, float]]:
    #    """Return docs most similar to query."""
    #    embedding = self._embedding.embed_query(query)
    #    documents = self.similarity_search_with_score_by_vector(
    #        embedding=embedding, k=k
    #    )
    #    return documents

    #def similarity_search_with_score_by_vector(
    #    self, embedding: List[float], k: int = 4, **kwargs: Any
    #) -> List[Tuple[Document, float]]:
    #    """Return docs most similar to query."""
    #    documents = self.similarity_search_with_score_by_vector(
    #        embedding=embedding, k=k
    #    )
    #    return documents

    @classmethod
    def from_texts(
        cls: VectorStore,#Type[SQLiteVec],
        titles: List[str],
        texts: List[str],
        embeds: List[List[float]],
        #embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        table: str = "langchain",
        db_file: str = "vec.db",
        lang: str = "en",
        **kwargs: Any,
    ) -> VectorStore:#SQLiteVec:
        """Return VectorStore initialized from texts and embeddings."""
        connection = cls.create_connection(db_file, lang)
        vec = cls(
            table=table, connection=connection, db_file=db_file, #embedding=embedding
        )
        vec.add_texts(titles=titles, texts=texts, embeds=embeds, metadatas=metadatas)
        return vec

    @staticmethod
    def create_connection(db_file: str, lang: str = "en") -> sqlite3.Connection:
        import sqlite3
        import sqlite_vec
        from sqlitefts import fts5

        connection = sqlite3.connect(db_file)
        connection.row_factory = sqlite3.Row
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)

        tokenizer = None
        if lang == "en":
            tokenizer = fts5.make_fts5_tokenizer(SpacyENTokenizer())
        elif lang == "de":
            tokenizer = fts5.make_fts5_tokenizer(SpacyDETokenizer())
        if tokenizer:
            fts5.register_tokenizer(connection, 'spacy_tokenizer', tokenizer)
        
        connection.text_factory = lambda b: b.decode(errors = 'ignore')
        return connection

    def get_dimensionality(self) -> int:
        """
        Function that does a dummy embedding to figure out how many dimensions
        this embedding function returns. Needed for the virtual table DDL.

        we use [`Qwen3-Embedding-8B-Q8_0.gguf`](https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF/blob/main/Qwen3-Embedding-8B-Q8_0.gguf) with last pooling
        """
        #dummy_text = "This is a dummy text"
        #dummy_embedding = self._embedding.embed_query(dummy_text)
        #return len(dummy_embedding)
        return 4096
