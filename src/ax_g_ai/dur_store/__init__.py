"""PostgreSQL(``dur`` 스키마, 말뭉치 스키마와 같은 DB)에서 서빙하는 심평원 DUR 목록 패키지.

설계상 위치: DUR 점검 데이터의 저장 계층. 스냅샷 빌드 CLI(``snapshot``), 적재 CLI(``loader``), 런타임 읽기(``reader``)로 나뉜다.

역할: 공공데이터 CSV → 스냅샷 → PostgreSQL 데이터셋 → 메모리 표(``DurTables``)로 이어지는 흐름을 담는다.
"""
