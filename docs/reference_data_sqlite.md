# 의약품·상병 참조 데이터 SQLite 변환

`medicine_dict.pkl`과 `disease_dict.pkl`을 조회 가능한 SQLite 데이터베이스로 변환한다. 원본 Pickle을 서비스 런타임에서 직접 읽지 않고, 변환 시점에만 기본 자료형으로 구성됐는지 검사한다.

## 실행

프로젝트 루트에서 다음을 실행한다.

```bash
uv run ax-g-ai-reference-db \
  --medicine medicine_dict.pkl \
  --disease disease_dict.pkl \
  --output data/reference_data.sqlite
```

출력 파일이 이미 있으면 변환기는 중단한다. 기존 파일을 교체해야 할 때만 명시적으로 `--replace`를 추가한다.

```bash
uv run ax-g-ai-reference-db --output data/reference_data.sqlite --replace
```

> **기본 비활성:** 원본 Pickle의 출처·라이선스가 확인되지 않았으므로 `AX_G_ATTRIBUTED_SOURCES_ONLY=true`(기본)에서는 아래 경로를 설정해도 쓰지 않는다(시작 로그 `reference_data_disabled`). 출처와 이용 조건(예: 공공데이터포털 이용허락범위)이 확인된 뒤 이 설정을 끄면 다시 쓰인다.

변환한 참조 정보를 환자 문맥 준비 시 보강하려면 서버 환경에 다음을 설정한다. 설정하지 않으면 기존 동작처럼 참조 보강 없이 실행된다.

```bash
AX_G_REFERENCE_DATA_DB_PATH=/absolute/path/to/reference_data.sqlite
```

설정된 DB에서는 처방의 `medCode`가 정확히 한 개의 약품 레코드와 일치할 때만 약품명, 효능·효과, 주의사항을 보강한다. Bridge가 선택적으로 전달한 `diagnosisList[].diagnosisCode`는 상병 코드와 한글 상병명을 연결한다. 용법·용량(`usage`)은 환자별 처방 근거로 사용하거나 AI에 전달하지 않는다.

## 생성 스키마

| 테이블 | 주요 열 | 용도 |
|---|---|---|
| `diseases` | `code`, `name_ko` | 상병 코드에서 한글 상병명 조회 |
| `medicines` | `source_kd_code`, `kd_code`, `name_ko`, `efficacy`, `usage`, `precautions` | 약품 코드에서 약품명·효능·용법·주의사항 조회 |
| `metadata` | `key`, `value` | 원본 파일명, SHA-256, 생성 시각, 스키마 버전 |

예시 조회:

```sql
SELECT name_ko FROM diseases WHERE code = 'A000';

SELECT name_ko, efficacy, precautions
FROM medicines
WHERE kd_code = '653600740';
```

## 운영 주의사항

- 원본 데이터의 발행일·코드 체계·개정 이력은 Pickle에 포함되지 않는다. 이 정보는 별도로 검증하고 관리한다.
- 원본 약품 CSV에서 `kd_code`가 비어 있어 Pickle 키가 `NaN`인 레코드는 약품을 식별할 수 없으므로 변환에서 제외한다.
- `source_kd_code`는 원본 키를 그대로 보존한다. `kd_code`는 양끝 공백을 제거한 검색용 코드이며, 원본 오류로 같은 `kd_code`가 여러 레코드에 연결될 수 있으므로 고유 키가 아니다.
- `usage`와 `precautions`는 참조 정보이며, 자동 처방·용량 결정에 직접 사용하지 않는다.
- 새 원본으로 변환하면 SHA-256이 `metadata`에 기록된다. 서비스 배포 전에 데이터 출처·승인 상태를 함께 검토한다.

## 질문의 참조 정보 검색

DB가 설정되면 준비된 환자 채팅의 일반 및 스트리밍 응답에서 질문에 **정확한 전체 상병명 또는 약품명**이 포함된 항목을 검색한다. 각 종류 최대 5건을 일반 참조 정보로 환자 EMR과 분리해 전달하며, 중복 약품 코드는 제외하고 효능과 주의사항은 각각 600자로 제한한다. 약어, 부분 이름, 증상이나 검사 결과를 통한 의미 검색은 지원하지 않는다. 기존 EMR 코드 보강은 그대로 적용된다.

참조에는 원본 파일명, 해시와 DB 생성 시각이 포함된다. 이는 원본의 발행일·임상 검증 또는 최신성을 뜻하지 않는다. 상병 테이블에는 진단 기준이 없으므로 상병명 자체를 진단 근거로 사용하지 않는다. 증상 기반 감별 진단의 외부 근거 검색은 출처와 버전이 검증된 진단 기준 자료를 별도로 확보하고 연동해야 한다.
