"""Corpus contract shared by the build output, the loaders and the runtime (standard library only).

- ``files``: file names of the generated corpus (``text_data/corpus``) and the default embedding model.
- ``gate``: ``is_citable``, the single Python gate every corpus record passes before the AI may use it.
- ``text``: the search tokenizer, shared by the in-memory index and the PostgreSQL ``search_tokens`` column.
"""
