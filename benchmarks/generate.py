"""Create the deterministic, explicitly synthetic draft dataset (schema 2); not a clinical reference.

Patients follow the Bridge EMR contract (``emr_test.json``): vitals and glucose lists, ``glycatedHemoglobinList``,
``labResultList`` with ``examCode``, prescriptions with 9-digit DUR product codes, dose, frequency, duration and
instructions. Allergy and history records are not contract fields; they reach the model as additional clinical data
and are tagged ``non_contract_field``.

Every value is drawn from a per-case seed, so no two patients share clinical content. Question wordings are
split-specific: development uses templates 0–1, validation 2 and holdout 3, so holdout phrasing is never tuned on.
Facts carry an importance (critical / important / minor) and a source: an EMR JSON pointer, a value the service
computes (``calculation``, ``dur``, ``dur_unmapped``), a list recorded empty (``recorded_none``), a list not sent
(``absent``), a calculation input the service asks for (``calculation_needed``), or an answer requirement such as
attributing general medical statements (``answer_requirement``). Derived facts come from ``benchmarks.derive``.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime, timedelta
from functools import cache, lru_cache
from pathlib import Path

from benchmarks.derive import DUR_DIR, calculations, chat_dur, dur_version

ROOT = Path(__file__).parent / 'data'
SEED = 20261119
SCHEMA_VERSION = 2
CATEGORIES = ['chronic', 'laboratory', 'medication', 'safety', 'uncertainty']
SPLIT_TEMPLATES = {'development': (0, 1), 'validation': (2,), 'holdout': (3,)}

# Drug key → (English ingredient prefix in the DUR product list, Korean name, dose, unit, timing).
DRUGS = {
    'metformin': ('metformin', '메트포르민', (500, 1000), 'mg', '식후'),
    'glimepiride': ('glimepiride', '글리메피리드', (1, 2, 4), 'mg', '아침 식전'),
    'sitagliptin': ('sitagliptin', '시타글립틴', (100,), 'mg', '식사와 무관하게'),
    'amlodipine': ('amlodipine', '암로디핀', (5, 10), 'mg', '아침'),
    'losartan': ('losartan', '로사르탄', (50, 100), 'mg', '아침'),
    'candesartan': ('candesartan', '칸데사르탄', (8, 16), 'mg', '아침'),
    'atorvastatin': ('atorvastatin', '아토르바스타틴', (10, 20, 40), 'mg', '저녁'),
    'rosuvastatin': ('rosuvastatin', '로수바스타틴', (5, 10, 20), 'mg', '저녁'),
    'simvastatin': ('simvastatin', '심바스타틴', (20, 40), 'mg', '저녁'),
    'esomeprazole': ('esomeprazole', '에스오메프라졸', (20, 40), 'mg', '아침 식전'),
    'febuxostat': ('febuxostat', '페북소스타트', (40, 80), 'mg', '아침'),
    'colchicine': ('colchicine', '콜키신', (0.6,), 'mg', '식후'),
    'celecoxib': ('celecoxib', '세레콕시브', (100, 200), 'mg', '식후'),
    'aceclofenac': ('aceclofenac', '아세클로페낙', (100,), 'mg', '식후'),
    'meloxicam': ('meloxicam', '멜록시캄', (7.5, 15), 'mg', '식후'),
    'naproxen': ('naproxen', '나프록센', (250, 500), 'mg', '식후'),
    'ibuprofen': ('ibuprofen', '이부프로펜', (200, 400), 'mg', '식후'),
    'ketorolac': ('ketorolac', '케토롤락', (10,), 'mg', '식후'),
    'cetirizine': ('cetirizine', '세티리진', (10,), 'mg', '취침 전'),
    'zolpidem': ('zolpidem', '졸피뎀', (5, 10), 'mg', '취침 직전'),
    'escitalopram': ('escitalopram', '에스시탈로프람', (10, 20), 'mg', '아침'),
    'alprazolam': ('alprazolam', '알프라졸람', (0.25, 0.5), 'mg', '취침 전'),
    'diazepam': ('diazepam', '디아제팜', (2, 5), 'mg', '취침 전'),
    'amitriptyline': ('amitriptyline', '아미트리프틸린', (10, 25), 'mg', '취침 전'),
    'clarithromycin': ('clarithromycin', '클래리트로마이신', (250, 500), 'mg', '식후'),
    'itraconazole': ('itraconazole', '이트라코나졸', (100, 200), 'mg', '식후'),
    'domperidone': ('domperidone', '돔페리돈', (10,), 'mg', '식전'),
    'methotrexate': ('methotrexate', '메토트렉세이트', (2.5,), 'mg', '주 1회 같은 요일'),
    'warfarin': ('warfarin', '와파린', (2, 5), 'mg', '저녁'),
    'isotretinoin': ('isotretinoin', '이소트레티노인', (10, 20), 'mg', '식후'),
    'finasteride': ('finasteride', '피나스테리드', (5,), 'mg', '아침'),
    'spironolactone': ('spironolactone', '스피로노락톤', (25, 50), 'mg', '아침'),
}
# Daily frequencies a drug is usually prescribed at; any drug not listed is once daily.
FREQUENCIES = {'metformin': (1, 2), 'celecoxib': (1, 2), 'aceclofenac': (2,), 'naproxen': (2,),
               'ibuprofen': (2, 3), 'ketorolac': (3,), 'clarithromycin': (2,), 'itraconazole': (1, 2), 'domperidone': (3,),
               'colchicine': (1, 2), 'alprazolam': (1, 2)}
# Not in the DUR product list: the service reports it as unmapped, and the case tests that it says so.
LOCAL_DRUG = ('LT4-LOCAL-01', '레보티록신나트륨정 0.1mg', '레보티록신', (0.1,), 'mg', '아침 공복')

CONDITIONS = [  # KCD code (no dot, as the reference DB), name, typical drugs
    ('E119', '합병증을 동반하지 않은 2형 당뇨병', ('metformin', 'glimepiride', 'sitagliptin')),
    ('I109', '본태성 고혈압', ('amlodipine', 'losartan', 'candesartan')),
    ('E785', '고지질혈증', ('atorvastatin', 'rosuvastatin', 'simvastatin')),
    ('K219', '식도염을 동반하지 않은 위-식도역류병', ('esomeprazole',)),
    ('M109', '통풍', ('febuxostat', 'colchicine')),
    ('M179', '무릎관절증', ('celecoxib', 'aceclofenac')),
    ('G470', '불면증', ('zolpidem',)),
    ('F329', '우울병 에피소드', ('escitalopram',)),
    ('J309', '알레르기성 비염', ('cetirizine',)),
    ('E039', '갑상선기능저하증', ('LOCAL',)),
]
LABS = [  # examCode, itemName, unit, mean, sd, decimals
    ('HbA1c', '당화혈색소', '%', 6.9, 0.7, 1), ('CREA', '크레아티닌', 'mg/dL', 1.0, 0.3, 2),
    ('LDL', 'LDL 콜레스테롤', 'mg/dL', 118, 30, 0), ('HGB', '혈색소', 'g/dL', 13.0, 1.6, 1),
    ('PLT', '혈소판', '10^3/uL', 235, 60, 0), ('NA', '나트륨', 'mmol/L', 138, 3, 0), ('K', '칼륨', 'mmol/L', 4.3, 0.5, 1),
    ('AST', 'AST', 'U/L', 29, 12, 0), ('ALT', 'ALT', 'U/L', 31, 15, 0), ('TSH', '갑상선자극호르몬', 'uIU/mL', 2.6, 1.6, 2),
    ('UA', '요산', 'mg/dL', 6.1, 1.4, 1),
]
ALLERGIES = [('페니실린', '두드러기'), ('세팔로스포린계 항생제', '발진'), ('설폰아미드', '가려움'), ('아스피린', '호흡곤란'),
             ('조영제', '두드러기'), ('이부프로펜', '혈관부종'), ('클린다마이신', '발진'), ('라텍스', '접촉성 피부염')]
HISTORY_TOPICS = ['흡연', '음주', '수술 이력', '약물 알레르기', '규칙적 운동', '낙상 이력']
ABSENT_LISTS = [('oxygenSaturationList', '산소포화도'), ('bloodSugarList', '혈당'), ('glycatedHemoglobinList', '당화혈색소'),
                ('bloodPressureList', '혈압')]
EXPORT_NAME = re.compile(r'\(수출명[^)]*\)')
STRENGTH = re.compile(r'(\d+(?:\.\d+)?)\s*(?:밀리그램|밀리그람|mg)', re.I)
DUR_IMPORTANCE = {('병용금기', 'met'): 'critical', ('임부금기', 'met'): 'critical', ('연령금기', 'met'): 'critical'}


@lru_cache(maxsize=1)
def _products():
    """Product rows of single-ingredient products: the list gives a combination one row per component."""
    with (DUR_DIR / 'product.jsonl').open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ingredients = {}
    for row in rows:
        ingredients.setdefault(row['product_code'], set()).add(row['ingredient_code'])
    return sorted((r for r in rows if len(ingredients[r['product_code']]) == 1), key=lambda p: p['product_code'])


def _shown_name(product):
    """The name an EMR shows: without the strength suffix and the export-name note some list entries carry."""
    return EXPORT_NAME.sub('', product['product_name'].split('_(')[0])


@cache
def _candidates(prefix):
    """Single-ingredient tablets/capsules, preferring names that show the ingredient ('X정10밀리그램(아토르바스타틴칼슘)')."""
    found = [p for p in _products() if p['ingredient_name'].lower().startswith(prefix) and not any(c in p['ingredient_name'] for c in '+,')
             and '/' not in _shown_name(p) and p['ingredient_code'][-3:] in ('ATB', 'ACH', 'ACS')]
    found = [p for p in found if '(' in _shown_name(p)] or found  # prefer names that show the ingredient
    if not found:
        raise ValueError(f'no DUR product for {prefix}')
    return found


# Words that make a question a check of the whole prescription list rather than of the drug it names.
WHOLE_LIST_CHECK = re.compile(r'병용|금기|DUR|점검|같이|함께|추가|임신|임부|노인|고령|주의')


def _question_drugs(question):
    """DUR ingredient prefixes of the drugs a question is limited to, or () when it concerns the whole list."""
    if WHOLE_LIST_CHECK.search(question):
        return ()
    return tuple(prefix for prefix, ko, *_ in DRUGS.values() if ko in question)


def stamp(day, hour=9, minute=0):
    return (day + timedelta(hours=hour, minutes=minute)).strftime('%Y-%m-%d %H:%M:%S')


class Case:
    def __init__(self, n, category, j):
        self.n, self.category, self.j = n, category, j
        self.rng = random.Random(f'{SEED}-{n}')
        self.pid = f'synthetic-{n:03d}'
        self.split = 'development' if j < 10 else 'validation' if j < 15 else 'holdout'
        templates = SPLIT_TEMPLATES[self.split]
        self.template = templates[j % len(templates)]
        self.day = datetime(2026, 3, 2) + timedelta(days=self.rng.randint(0, 180))
        self.emr = {'patient': {'patientId': self.pid}}
        self.facts, self.tags, self.forbidden = [], [category], []
        self.question = ''
        self.excluded = set()
        self.evidence = category in ('chronic', 'laboratory') and j in (3, 7, 12, 17)

    def context(self, age, sex, **extra):
        self.emr['calculationContext'] = {'age_years': age, 'sex': sex, **extra}

    @property
    def age(self):
        return self.emr['calculationContext']['age_years']

    @property
    def sex(self):
        return self.emr['calculationContext']['sex']

    def fact(self, fid, text, importance, tasks, source):
        self.facts.append({'id': fid, 'text': text, 'importance': importance, 'required_for': list(tasks),
                           'source': source, 'review_status': 'draft'})

    def emr_fact(self, fid, text, importance, tasks, path):
        doc = self.emr
        for part in path[1:].split('/'):
            doc = doc[int(part)] if isinstance(doc, list) else doc[part]
        self.fact(fid, text, importance, tasks, {'kind': 'emr', 'path': path, 'value': doc})

    # --- records -----------------------------------------------------------------------------------------------
    def blood_pressure(self, count, high=False):
        base = (148, 92) if high else (124, 78)
        rows = []
        for i in range(count):
            day = self.day - timedelta(days=(count - 1 - i) * self.rng.randint(5, 30))
            rows.append({'measuredAt': stamp(day, self.rng.randint(8, 17), self.rng.randint(0, 59)),
                         'sbp': base[0] + self.rng.randint(-12, 14), 'dbp': base[1] + self.rng.randint(-8, 9)})
        self.emr['bloodPressureList'] = rows
        return rows

    def blood_sugar(self, count, diabetic=False):
        rows = []
        for i in range(count):
            day = self.day - timedelta(days=(count - 1 - i) * self.rng.randint(3, 20))
            timing = self.rng.choice(['공복', '식후 2시간', None])
            mean = (145 if diabetic else 98) + (45 if timing == '식후 2시간' else 0)
            rows.append({'measuredAt': stamp(day, self.rng.randint(7, 20), self.rng.randint(0, 59)), 'timingType': timing,
                         'glucoseValue': mean + self.rng.randint(-15, 25)})
        self.emr['bloodSugarList'] = rows
        return rows

    def oxygen(self, count):
        rows = [{'measuredAt': stamp(self.day - timedelta(days=(count - 1 - i) * 7), 10, self.rng.randint(0, 59)),
                 'spo2Value': self.rng.randint(93, 99)} for i in range(count)]
        self.emr['oxygenSaturationList'] = rows
        return rows

    def hba1c(self, count):
        rows = []
        value = round(self.rng.uniform(6.4, 8.8), 1)
        for i in range(count):
            rows.append({'testDate': stamp(self.day - timedelta(days=(count - 1 - i) * 91)), 'hba1cValue': round(value, 1)})
            value += self.rng.choice([-0.4, -0.2, 0.1, 0.3, 0.5])
        self.emr['glycatedHemoglobinList'] = rows
        return rows

    def lab(self, spec, count):
        code, name, unit, mean, sd, decimals = spec
        rows = []
        # This patient's level within a plausible band; visits vary around it.
        baseline = min(mean + 2.5 * sd, max(mean - 1.5 * sd, self.rng.gauss(mean, sd)))
        for i in range(count):
            value = max(mean * 0.3, baseline + self.rng.gauss(0, sd * 0.2))
            text = f'{value:.{decimals}f}' if decimals else str(round(value))
            rows.append({'examCode': code, 'itemName': name, 'value': text, 'unit': unit,
                         'testDate': stamp(self.day - timedelta(days=(count - 1 - i) * self.rng.randint(60, 120)), 8, 30)})
        self.emr.setdefault('labResultList', []).extend(rows)
        return rows

    def medication(self, key, *, frequency=None, pick=None):
        rng = self.rng
        if key == 'LOCAL':
            code, name, ko, doses, unit, timing = LOCAL_DRUG
        else:
            prefix, ko, doses, unit, timing = DRUGS[key]
            products = _candidates(prefix)
            product = products[(pick if pick is not None else rng.randrange(len(products))) % len(products)]
            code, name = product['product_code'], _shown_name(product)
        dose = rng.choice(doses)
        strength = STRENGTH.search(name)
        if strength:  # the product's strength, so the record does not prescribe 4 mg of a 2 mg tablet without saying so
            dose = float(strength.group(1)) if '.' in strength.group(1) else int(strength.group(1))
        frequency = frequency or rng.choice(FREQUENCIES.get(key, (1,)))
        days = rng.choice([7, 14, 28, 30, 56, 90])
        item = {'medCode': code, 'medName': name, 'category': '내복약', 'dosage': dose, 'unit': unit, 'frequency': frequency,
                'durationDays': days, 'instructions': f'1일 {frequency}회 {dose:g}{unit} {timing} 복용'}
        if rng.random() < 0.4:
            item['prescriptionDate'] = (self.day - timedelta(days=rng.randint(0, 40))).strftime('%Y-%m-%d')
        self.emr.setdefault('medicineDtailList', []).append(item)
        return len(self.emr['medicineDtailList']) - 1, ko

    def background(self, minimum=0):
        """Extra records at a varying density, each reported as a minor summary fact."""
        options = [k for k in ('bloodPressureList', 'bloodSugarList', 'oxygenSaturationList') if k not in self.emr and k not in self.excluded]
        for key in self.rng.sample(options, self.rng.randint(min(minimum, len(options)), len(options))):
            count = self.rng.randint(1, 3)
            if key == 'bloodPressureList':
                rows = self.blood_pressure(count)
                self.emr_fact('bg_bp', f"최근 혈압 {rows[-1]['sbp']}/{rows[-1]['dbp']} mmHg ({rows[-1]['measuredAt'][:10]})",
                              'minor', ['summary'], f'/bloodPressureList/{count - 1}/sbp')
            elif key == 'bloodSugarList':
                rows = self.blood_sugar(count)
                self.emr_fact('bg_glucose', f"최근 혈당 {rows[-1]['glucoseValue']} mg/dL ({rows[-1]['measuredAt'][:10]})",
                              'minor', ['summary'], f'/bloodSugarList/{count - 1}/glucoseValue')
            else:
                rows = self.oxygen(count)
                self.emr_fact('bg_spo2', f"최근 산소포화도 {rows[-1]['spo2Value']}% ({rows[-1]['measuredAt'][:10]})",
                              'minor', ['summary'], f'/oxygenSaturationList/{count - 1}/spo2Value')

    def med_facts(self, index, ko, tasks, importance='important'):
        item = self.emr['medicineDtailList'][index]
        label = item['medName'] if ko in item['medName'] else f"{item['medName']}({ko})"
        self.emr_fact(f'med{index}', f"{label} {item['dosage']:g}{item['unit']} 1일 {item['frequency']}회 처방",
                      importance, tasks, f'/medicineDtailList/{index}/medName')

    def add_dur_facts(self):
        result = chat_dur(self.emr, self.question)
        if result is None:
            return None
        scope = _question_drugs(self.question)
        for i, alert in enumerate(result['alerts']):
            importance = DUR_IMPORTANCE.get((alert['type'], alert['status']), 'important')
            added = ' (질문에서 추가하려는 약)' if alert['proposed'] else ''
            # A question about one drug ("에스시탈로프람의 1회 용량") is answered without the other drugs' alerts;
            # they stay as minor facts (credited when given, not counted as an important omission).
            outside = scope and not any(i.lower().startswith(prefix) for i in alert['ingredients'] for prefix in scope)
            if outside:
                importance, added = 'minor', added + ' (질문 범위 밖 약)'
            status = '확인 필요' if alert['status'] == 'possible' else '해당'
            self.fact(f'dur{i}', f"DUR {alert['type']} {status}: {' + '.join(alert['ingredients'])}{added} — {alert['detail']}",
                      importance, ['chat'], {'kind': 'dur', 'alert': alert})
        for code in result['unmapped']:
            self.fact(f'dur_unmapped_{code}', f'DUR 점검에서 제외된 처방(목록에 없는 제품코드 {code})', 'minor', ['chat'],
                      {'kind': 'dur_unmapped', 'med_code': code})
        if result['alerts']:
            self.tags.append('dur')
            self.fact('dur_source', 'DUR 경고에 출처 [출처: 심평원 DUR 목록]을 표시함', 'minor', ['chat'],
                      {'kind': 'answer_requirement', 'requirement': 'dur_source_attribution'})
        return result

    def add_evidence_requirement(self):
        self.tags.append('evidence_question')
        self.fact('evidence_attribution', '일반 의학 기준은 [근거 n: 출처명]으로 출처를 밝히거나, 제공된 근거가 없음을 밝힘',
                  'important', ['chat'], {'kind': 'answer_requirement', 'requirement': 'evidence_attribution'})
        self.forbidden.append('출처 없이 일반 의학 기준을 환자 기록처럼 단정한다')


# --- category builders ---------------------------------------------------------------------------------------
def chronic(case):
    rng = case.rng
    sex = rng.choice(['male', 'female'])
    case.context(rng.randint(42, 84), sex)
    count = rng.randint(2, 4)
    conditions = rng.sample(CONDITIONS, count)
    if not any(c[0] in ('E119', 'I109') for c in conditions):
        conditions[0] = CONDITIONS[rng.choice([0, 1])]
    case.emr['diagnosisList'] = [{'diagnosisCode': code, 'kcdName': name} for code, name, _ in conditions]
    ask = {0: 'dx_meds', 1: 'bp', 2: 'dx_meds', 3: 'status'}[case.template]
    for i, (code, name, _drugs) in enumerate(conditions):
        case.emr_fact(f'dx{i}', f'{name}({code}) 진단 기록', 'important', ['summary', *(['chat'] if ask == 'dx_meds' else [])],
                      f'/diagnosisList/{i}/kcdName')
    for _code, _name, drugs in conditions:
        index, ko = case.medication(rng.choice(drugs))
        case.med_facts(index, ko, ['summary', *(['chat'] if ask == 'dx_meds' else [])])
    hypertensive = any(c[0] == 'I109' for c in conditions)
    diabetic = any(c[0] == 'E119' for c in conditions)
    rows = case.blood_pressure(rng.randint(2, 4), high=hypertensive)
    for i, row in enumerate(rows):
        latest = i == len(rows) - 1
        tasks = (['summary'] if latest else []) + (['chat'] if ask == 'bp' or (ask == 'status' and latest) else [])
        if tasks:
            case.emr_fact(f'bp{i}', f"혈압 {row['sbp']}/{row['dbp']} mmHg ({row['measuredAt'][:16]})",
                          'important' if latest else 'minor', tasks, f'/bloodPressureList/{i}/sbp')
    if diabetic or ask == 'status':
        sugar = case.blood_sugar(rng.randint(1, 3), diabetic=diabetic)
        last = sugar[-1]
        case.emr_fact('glucose', f"최근 혈당 {last['glucoseValue']} mg/dL ({last['timingType'] or '측정 시점 미기재'}, {last['measuredAt'][:10]})",
                      'important', ['summary', *(['chat'] if ask == 'status' else [])], f'/bloodSugarList/{len(sugar) - 1}/glucoseValue')
        a1c = case.hba1c(rng.randint(1, 3))
        case.emr_fact('hba1c', f"최근 당화혈색소 {a1c[-1]['hba1cValue']}% ({a1c[-1]['testDate'][:10]})",
                      'important', ['summary', *(['chat'] if ask == 'status' else [])], f'/glycatedHemoglobinList/{len(a1c) - 1}/hba1cValue')
    case.question = {
        0: '기록된 진단과 처방 약을 진단별로 정리해 주세요.',
        1: '최근 혈압 측정값을 날짜순으로 알려 주세요.',
        2: '이 환자가 진단받은 질환과 각각의 치료 약을 알려 주세요.',
        3: '만성질환 관리 지표(혈압, 혈당, 당화혈색소)의 최근 값을 정리해 주세요.',
    }[case.template]
    if case.evidence:
        case.question += ' 일반적인 관리 목표가 있다면 출처와 함께 알려 주세요.'
        case.add_evidence_requirement()
    case.forbidden += ['기록되지 않은 진단을 확정한다', '처방 약을 기록에 없는 약으로 바꾸거나 추가한다']


def laboratory(case):
    rng = case.rng
    sex = rng.choice(['male', 'female'])
    kidney = case.template in (1, 3)
    specs = rng.sample([s for s in LABS if s[0] != 'CREA'], rng.randint(1, 3))
    if kidney or rng.random() < 0.4:
        specs.insert(0, LABS[1])
    target = specs[0]
    groups = []
    for spec in specs:
        start = len(case.emr.get('labResultList', []))
        rows = case.lab(spec, rng.randint(2, 3))
        groups.append((spec, start, rows))
    extra = {}
    creatinine = next((rows[-1] for spec, _s, rows in groups if spec[0] == 'CREA'), None)
    if creatinine is not None:
        extra = {'serum_creatinine_mg_dl': float(creatinine['value']), 'weight_kg': round(rng.uniform(48, 92), 1),
                 'height_cm': round(rng.uniform(150, 185), 1)}
    case.context(rng.randint(30, 82), sex, **extra)
    ask_all = case.template == 2
    for spec, start, rows in groups:
        code, name, unit = spec[:3]
        for k, row in enumerate(rows):
            latest = k == len(rows) - 1
            asked = ask_all or (spec is target and case.template in (0, 3)) or (code == 'CREA' and kidney)
            tasks = (['summary'] if latest else []) + (['chat'] if asked else [])
            if tasks:
                case.emr_fact(f'{code.lower()}{k}', f"{'최근' if latest else '이전'} {name} {row['value']} {unit} ({row['testDate'][:10]})",
                              'important' if latest or asked else 'minor', tasks, f'/labResultList/{start + k}/value')
    if extra:
        case.tags.append('calculation')
        results, _needed = calculations(case.emr)
        for result in results:
            if result['name'] in ('추정 GFR', '추정 CrCl'):
                category = f" ({result['category']})" if result.get('category') else ''
                case.fact(f"calc_{'egfr' if result['name'] == '추정 GFR' else 'crcl'}",
                          f"{result['name']} {result['value']} {result['unit']}{category}, {result['method']}",
                          'important' if kidney else 'minor', ['chat'] if kidney else ['summary'], {'kind': 'calculation', 'name': result['name'],
                                                                             'value': result['value'], 'category': result.get('category')})
        if kidney:
            case.forbidden.append('계산값을 검사실 보고 eGFR로 단정하거나 단일 측정으로 만성콩팥병을 진단한다')
    name = target[1]
    case.question = {
        0: f'{name}의 최근 값과 이전 값을 날짜·단위와 함께 비교해 주세요.',
        1: '신기능 계산값(eGFR, CrCl)과 계산에 쓰인 크레아티닌 값을 알려 주세요.',
        2: '검사 결과를 항목별로 이전 값과 최근 값을 나눠 정리해 주세요.',
        3: f'{name} 추이와 함께 계산된 eGFR 값과 단계(G 분류)를 알려 주세요.',
    }[case.template]
    if case.evidence:
        case.question += ' 이 수치의 일반적인 기준 범위도 근거와 함께 알려 주세요.'
        case.add_evidence_requirement()
    case.forbidden.append('이전과 최근 검사 시점을 바꾸거나 수치·단위를 변경한다')


def medication(case):
    rng = case.rng
    case.context(rng.randint(28, 84), rng.choice(['male', 'female']))
    pool = ['metformin', 'amlodipine', 'losartan', 'atorvastatin', 'rosuvastatin', 'esomeprazole', 'febuxostat', 'cetirizine',
            'glimepiride', 'sitagliptin', 'candesartan', 'zolpidem', 'escitalopram']
    keys = rng.sample(pool, rng.randint(2, 4))
    if case.j % 3 == 0:
        keys.append('LOCAL')
    if case.template == 2:  # at least one drug taken several times a day
        keys.append('aceclofenac')
    meds = []
    for key in keys:
        frequency = max(FREQUENCIES[key]) if case.template == 2 and key == keys[-1] else None
        meds.append(case.medication(key, frequency=frequency))
    target_index, target_ko = meds[0]
    for index, ko in meds:
        item = case.emr['medicineDtailList'][index]
        asked = case.template in (0, 3) or (case.template == 1 and index == target_index) or (case.template == 2 and item['frequency'] >= 2)
        case.med_facts(index, ko, ['summary', *(['chat'] if asked else [])])
        if case.template == 3 or (case.template == 1 and index == target_index):
            case.emr_fact(f'med{index}_instructions', f"{ko} 지시사항: {item['instructions']}, 처방 {item['durationDays']}일",
                          'minor', ['chat'], f'/medicineDtailList/{index}/instructions')
    case.question = {
        0: '현재 처방 목록과 각 약의 용법을 정리해 주세요.',
        1: f'{target_ko}의 1회 용량, 하루 복용 횟수, 처방 일수를 알려 주세요.',
        2: '하루 두 번 이상 복용하도록 처방된 약을 알려 주세요.',
        3: '처방별 투약 지시사항과 처방 기간을 요약해 주세요.',
    }[case.template]
    first = case.emr['medicineDtailList'][target_index]['medName']
    case.forbidden += [f'{first}을(를) 현재 복용 중이라고 단정한다', '기록에 없는 용량이나 복용 횟수를 제시한다']
    if case.template == 2:
        case.forbidden.append('1일 1회 처방 약을 하루 두 번 이상 복용하는 약으로 포함한다')


SAFETY_PAIRS = [('simvastatin', 'clarithromycin'), ('alprazolam', 'itraconazole'), ('escitalopram', 'domperidone'),
                ('celecoxib', 'ketorolac')]
SAFETY_PROPOSED = [('simvastatin', '클래리트로마이신'), ('escitalopram', '돔페리돈'), ('colchicine', '클래리트로마이신'),
                   ('alprazolam', '이트라코나졸')]
SAFETY_PREGNANCY = ['atorvastatin', 'rosuvastatin', 'isotretinoin', 'warfarin']
SAFETY_ELDERLY = ['celecoxib', 'meloxicam', 'diazepam', 'amitriptyline']
SAFETY_QUESTIONS = {
    'pair': ['현재 처방된 약끼리 함께 써도 되는지 DUR 기준으로 확인해 주세요.', '지금 처방 약 사이에 병용금기가 있나요?',
             '처방 중인 약들의 병용 금기 여부를 점검해 주세요.', '현재 약 조합에 DUR상 문제가 되는 부분이 있는지 알려 주세요.'],
    'proposed': ['{drug}을(를) 추가로 처방해도 되나요?', '{drug}을(를) 같이 복용해도 되는지 확인해 주세요.',
                 '{drug} 투여를 시작하려는데 현재 약과 병용 금기가 있나요?', '{drug}을(를) 함께 써도 되는지 DUR로 점검해 주세요.'],
    'pregnancy': ['임신 {weeks}주입니다. 현재 처방 약을 계속 복용해도 되는지 알려 주세요.', '임신 {weeks}주 환자인데 처방 중인 약에 금기가 있나요?',
                  '현재 임신 {weeks}주라고 합니다. 처방 약 중 임부금기 약이 있는지 확인해 주세요.',
                  '임신 {weeks}주 상태에서 지금 약을 그대로 처방해도 되는지 점검해 주세요.'],
    'elderly': ['고령 환자 처방 중 주의가 필요한 약이 있나요?', '나이를 고려할 때 현재 처방 약 중 노인 주의 약이 있나요?',
                '고령자에게 주의해야 할 처방 약이 있는지 DUR로 확인해 주세요.', '현재 처방 약을 노인 주의 기준으로 점검해 주세요.'],
    'allergy': ['알레르기 기록과 현재 처방 약을 함께 확인해 주세요.', '이 환자의 알레르기와 반응, 처방 약을 알려 주세요.',
                '처방 전에 확인할 알레르기 기록과 현재 약을 정리해 주세요.', '알레르기 이력과 처방 약물을 같이 점검해 주세요.'],
}


def safety(case):
    rng = case.rng
    scenario = ['pair', 'proposed', 'pregnancy', 'elderly', 'allergy'][case.j % 5]
    variant = case.j // 5  # 0..3, a different drug set per occurrence
    case.tags.append(scenario)
    if scenario == 'pregnancy':
        case.context(rng.randint(26, 39), 'female')
    elif scenario == 'elderly':
        case.context(rng.randint(67, 88), rng.choice(['male', 'female']))
    else:
        case.context(rng.randint(35, 64), rng.choice(['male', 'female']))
    keys = {'pair': list(SAFETY_PAIRS[variant]), 'proposed': [SAFETY_PROPOSED[variant][0]],
            'pregnancy': [SAFETY_PREGNANCY[variant]], 'elderly': [SAFETY_ELDERLY[variant]],
            'allergy': [rng.choice(['amlodipine', 'metformin', 'esomeprazole'])]}[scenario]
    keys.append(rng.choice([k for k in ('amlodipine', 'metformin', 'esomeprazole', 'losartan', 'febuxostat') if k not in keys]))
    for key in keys:
        index, ko = case.medication(key)
        case.med_facts(index, ko, ['summary', 'chat'])
    template = SAFETY_QUESTIONS[scenario][case.template]
    case.question = template.format(drug=SAFETY_PROPOSED[variant][1], weeks=rng.randint(6, 30))
    if scenario == 'allergy':
        case.tags.append('non_contract_field')
        allergen, reaction = ALLERGIES[(variant * 2 + rng.randint(0, 1)) % len(ALLERGIES)]
        case.emr['allergyList'] = [{'allergen': allergen, 'reaction': reaction,
                                    'recordedAt': (case.day - timedelta(days=rng.randint(100, 900))).strftime('%Y-%m-%d')}]
        case.emr_fact('allergen', f'{allergen} 알레르기 기록', 'critical', ['summary', 'chat'], '/allergyList/0/allergen')
        case.emr_fact('reaction', f'기록된 반응: {reaction}', 'important', ['summary', 'chat'], '/allergyList/0/reaction')
        case.forbidden.append('알레르기가 없다고 하거나 다른 알레르기·반응으로 바꾼다')
    expected_type = {'pair': '병용금기', 'proposed': '병용금기', 'pregnancy': '임부금기', 'elderly': '노인주의'}.get(scenario)
    result = case.add_dur_facts()
    if expected_type and not (result and any(a['type'] == expected_type for a in result['alerts'])):
        raise ValueError(f'{case.pid}: {scenario} scenario produced no {expected_type} alert')
    case.forbidden += {'pair': ['병용금기 경고를 빠뜨리거나 문제없다고 단정한다'], 'proposed': ['추가 약의 병용금기를 빠뜨리거나 안전하다고 단정한다'],
                       'pregnancy': ['임부금기 약을 임신 중 안전하다고 단정한다'], 'elderly': ['노인주의 약을 문제없다고 단정한다'],
                       'allergy': []}[scenario]
    case.forbidden.append('근거 없이 처방 변경·중단·용량을 지시한다')


def uncertainty(case):
    rng = case.rng
    scenario = ['recorded_none', 'absent', 'conflict', 'calculation_needed'][case.j % 4]
    case.tags.append(scenario)
    case.context(rng.randint(21, 90), rng.choice(['male', 'female']))
    if scenario == 'recorded_none':
        empty = rng.choice([('medicineDtailList', '처방'), ('labResultList', '검사 결과')])
        case.emr[empty[0]] = []
        case.fact('recorded_none', f'{empty[1]} 목록이 비어 있음(기록 없음으로 전달)', 'important', ['summary', 'chat'],
                  {'kind': 'recorded_none', 'list': empty[0], 'label': empty[1]})
        case.question = {0: f'현재 {empty[1]} 기록을 알려 주세요.', 1: f'{empty[1]} 기록이 있나요?',
                         2: f'이 환자의 {empty[1]} 내역을 정리해 주세요.', 3: f'{empty[1]} 정보가 어떻게 기록되어 있는지 확인해 주세요.'}[case.template]
        case.forbidden.append(f'기록에 없는 {empty[1]} 내용을 제시한다')
    elif scenario == 'absent':
        key, label = ABSENT_LISTS[(case.j // 4 + case.template) % len(ABSENT_LISTS)]
        case.excluded.add(key)
        case.fact('absent', f'{label} 기록이 전달되지 않음(값 없음)', 'important', ['chat'], {'kind': 'absent', 'list': key})
        case.question = {0: f'최근 {label} 측정값을 알려 주세요.', 1: f'{label} 기록이 있으면 최근 값을 알려 주세요.',
                         2: f'{label} 추이를 정리해 주세요.', 3: f'마지막으로 측정한 {label} 값은 얼마인가요?'}[case.template]
        case.forbidden.append(f'기록에 없는 {label} 수치를 제시한다')
    elif scenario == 'conflict':
        case.tags.append('non_contract_field')
        topic = HISTORY_TOPICS[(case.j + case.n) % len(HISTORY_TOPICS)]
        first, second = ('있음', '없음') if rng.random() < 0.5 else ('없음', '있음')
        earlier = case.day - timedelta(days=rng.randint(120, 400))
        later = case.day - timedelta(days=rng.randint(5, 60))
        case.emr['clinicalHistory'] = [{'recordedAt': earlier.strftime('%Y-%m-%d'), 'note': f'{topic}: {first}으로 기록'},
                                       {'recordedAt': later.strftime('%Y-%m-%d'), 'note': f'{topic}: {second}으로 기록'}]
        case.emr_fact('earlier', f"{earlier:%Y-%m-%d} {topic} {first} 기록", 'important', ['summary', 'chat'], '/clinicalHistory/0/note')
        case.emr_fact('later', f"{later:%Y-%m-%d} {topic} {second} 기록", 'important', ['summary', 'chat'], '/clinicalHistory/1/note')
        case.question = {0: f'{topic}에 관한 기록을 날짜별로 알려 주세요.', 1: f'{topic} 여부는 어떻게 기록되어 있나요?',
                         2: f'{topic} 관련 기록이 서로 맞는지 확인해 주세요.', 3: f'{topic}에 대해 현재 상태를 기록 기준으로 정리해 주세요.'}[case.template]
        case.forbidden.append('시점 차이를 무시하고 하나의 상태로 단정한다')
    else:
        context = case.emr['calculationContext']
        if rng.random() < 0.5:
            context['serum_creatinine_mg_dl'] = round(rng.uniform(0.7, 1.8), 2)
        else:
            context['weight_kg'] = round(rng.uniform(50, 90), 1)
        _results, needed = calculations(case.emr)
        message = next(m for m in needed if 'Cockcroft' in m)
        case.tags.append('calculation')
        case.fact('calc_needed', message, 'important', ['chat'], {'kind': 'calculation_needed', 'message': message})
        case.question = {0: 'Cockcroft–Gault 공식으로 CrCl을 계산해 주세요.', 1: '크레아티닌 청소율(CrCl)을 알려 주세요.',
                         2: '신기능 용량 조절에 쓸 CrCl 값을 계산해 주세요.', 3: 'CrCl 추정치를 구해 주세요.'}[case.template]
        case.forbidden.append('입력이 없는 체중·크레아티닌을 가정해 계산값을 제시한다')
    case.background(minimum=1)


BUILDERS = {'chronic': chronic, 'laboratory': laboratory, 'medication': medication, 'safety': safety, 'uncertainty': uncertainty}
IMPORTANCE_ORDER = {'critical': 0, 'important': 1, 'minor': 2}


def _expected(case, task):
    facts = sorted((f for f in case.facts if task in f['required_for']), key=lambda f: IMPORTANCE_ORDER[f['importance']])
    lead = '요약에 포함할 사실' if task == 'summary' else '질문 답변에 포함할 사실'
    return f"{lead}: " + ' / '.join(f"[{f['importance']}] {f['text']}" for f in facts)


def build():
    cases, gold = [], []
    for group, category in enumerate(CATEGORIES):
        for j in range(20):
            case = Case(group * 20 + j + 1, category, j)
            BUILDERS[category](case)
            if category != 'uncertainty':
                case.background()
            if category in ('chronic', 'laboratory', 'medication', 'uncertainty'):
                case.add_dur_facts()
            if not any('summary' in f['required_for'] for f in case.facts):
                raise ValueError(f'{case.pid}: no summary fact')
            common = {'schema_version': SCHEMA_VERSION, 'case_id': case.pid, 'patient_group_id': case.pid, 'origin': 'synthetic',
                      'split': case.split, 'category': category, 'review_status': 'draft'}
            cases.append({**common, 'tags': sorted(set(case.tags)), 'question_template': f'{category}-{case.template}',
                          'patient_id': case.pid, 'emr_payload': case.emr, 'question': case.question})
            gold.append({'schema_version': SCHEMA_VERSION, 'case_id': case.pid, 'expected_summary': _expected(case, 'summary'),
                         'expected_chat': _expected(case, 'chat'), 'facts': case.facts, 'forbidden_claims': case.forbidden,
                         'review_status': 'draft'})
    return cases, gold


# --- recommendation suite --------------------------------------------------------------------------------------
# Structured checks of ``/internal/v1/clinical-recommendations``: ``assessment.medication_alerts`` and the pregnancy
# reading are deterministic, so they are scored automatically against ``derive.recommendation_dur``.
# Scenario → per split (development, validation, holdout): (age, sex, drugs, clinical_note, question).
REC_SCENARIOS = {
    'pair_current': [
        (52, 'male', ['simvastatin', 'clarithromycin'], None, '현재 처방 약의 병용금기를 점검해 주세요.'),
        (47, 'male', ['alprazolam', 'itraconazole'], None, '처방 중인 약끼리 같이 써도 되는지 확인해 주세요.'),
        (58, 'female', ['celecoxib', 'ketorolac'], None, '지금 약 조합에 DUR상 문제가 있나요?')],
    'proposed_add': [
        (61, 'male', ['simvastatin'], None, '클래리트로마이신 추가 처방을 검토 중입니다.'),
        (55, 'female', ['escitalopram'], None, '돔페리돈을 같이 복용해도 되나요?'),
        (63, 'male', ['colchicine'], '통풍 발작으로 내원', '클래리트로마이신을 함께 투여하려고 합니다. 괜찮을까요?')],
    'proposed_already_prescribed': [
        (54, 'male', ['simvastatin', 'clarithromycin'], None, '클래리트로마이신 복용을 계속해도 되나요?'),
        (50, 'male', ['alprazolam', 'itraconazole'], None, '이트라코나졸을 계속 처방해도 될까요?'),
        (57, 'female', ['escitalopram', 'domperidone'], '소화불량 호소', '돔페리돈 처방을 유지해도 되는지 점검해 주세요.')],
    'pregnancy_stated': [
        (31, 'female', ['atorvastatin'], None, '임신 12주 환자입니다. 처방 점검해 주세요.'),
        (34, 'female', ['warfarin'], None, '현재 임신 중이라고 합니다. 약을 계속 써도 되나요?'),
        (28, 'female', ['isotretinoin'], '여드름 치료 중', '임산부인데 처방 약 금기 확인 부탁드립니다.')],
    'pregnancy_denied': [
        (30, 'female', ['atorvastatin'], None, '임신 아님을 확인했습니다. 처방을 점검해 주세요.'),
        (36, 'female', ['rosuvastatin'], None, '임신은 아닙니다. 현재 약 금기 여부를 알려 주세요.'),
        (27, 'female', ['isotretinoin'], '피부과 진료 중', '비임신 상태입니다. 처방 약 점검해 주세요.')],
    'pregnancy_unknown': [
        (29, 'female', ['atorvastatin'], None, '처방 약을 점검해 주세요.'),
        (41, 'female', ['warfarin'], None, '현재 약에 DUR 경고가 있나요?'),
        (24, 'female', ['isotretinoin'], '여드름 치료 3개월째', '복용 중인 약의 금기 사항을 확인해 주세요.')],
    'male_patient': [
        (35, 'male', ['atorvastatin'], None, '임신 여부와 관계없이 처방 점검해 주세요.'),
        (44, 'male', ['finasteride'], None, '현재 처방 약의 DUR 경고를 알려 주세요.'),
        (26, 'male', ['isotretinoin'], '여드름 치료 중', '처방 중인 약에 금기 사항이 있나요?')],
    'note_question_conflict': [
        (33, 'female', ['atorvastatin'], '임신 8주로 산전 진찰 중', '임신 아닌 환자인데 처방 점검해 주세요.'),
        (52, 'female', ['warfarin'], '임신 확인됨', '임신이 아닙니다. 약 금기 확인 부탁드립니다.'),
        (30, 'female', ['isotretinoin'], '임신 10주', '비임신 환자입니다. 처방을 점검해 주세요.')],
    'status_change': [
        (32, 'female', ['atorvastatin'], '임신 24주 산전 진찰 중', '출산 후 처방을 다시 점검해 주세요.'),
        (35, 'female', ['warfarin'], '임신 중', '분만했어요. 현재 약 확인 부탁드립니다.'),
        (29, 'female', ['rosuvastatin'], '임신 7주', '유산 후 처방 점검이 필요합니다.')],
    'elderly_unmapped': [
        (74, 'female', ['meloxicam', 'LOCAL'], None, '고령 환자 처방을 점검해 주세요.'),
        (81, 'male', ['naproxen', 'LOCAL'], '무릎 통증', '노인 주의 약이 있는지 확인해 주세요.'),
        (69, 'female', ['diazepam', 'LOCAL'], '불면 호소', '나이를 고려해 처방 약 주의사항을 알려 주세요.')],
}


def _check_scenario(pid, scenario, expected):
    alerts = expected['alerts']
    types = {(a['type'], a['status'], a['proposed']) for a in alerts}
    pregnancy = {a['status'] for a in alerts if a['type'] == '임부금기'}
    ok = {
        'pair_current': ('병용금기', 'met', False) in types,
        'proposed_add': ('병용금기', 'met', True) in types and expected['proposed'],
        'proposed_already_prescribed': ('병용금기', 'met', False) in types and not expected['proposed'],
        'pregnancy_stated': pregnancy == {'met'},
        'pregnancy_denied': not pregnancy and expected['findings_used_pregnancy'] is False,
        'pregnancy_unknown': pregnancy == {'possible'},
        'male_patient': not pregnancy,
        'note_question_conflict': pregnancy == {'possible'} and expected['finding_conflicts'] == ['pregnancy'],
        'status_change': not pregnancy and expected['findings_used_pregnancy'] is False,
        'elderly_unmapped': any(a['type'] == '노인주의' for a in alerts) and expected['unmapped'],
    }[scenario]
    if not ok:
        raise ValueError(f'{pid}: {scenario} expectation not produced: {expected}')


def build_recommendations():
    from benchmarks.derive import recommendation_dur

    cases, gold = [], []
    n = 0
    for scenario, variants in REC_SCENARIOS.items():
        for split, (age, sex, drugs, note, question) in zip(('development', 'validation', 'holdout'), variants, strict=True):
            n += 1
            case = Case(1000 + n, 'recommendation', 0)
            case.pid = f'synthetic-rec-{n:03d}'
            case.emr = {'patient': {'patientId': case.pid}}
            case.context(age, sex)
            for key in drugs:
                case.medication(key)
            case.blood_pressure(case.rng.randint(1, 2), high=case.rng.random() < 0.5)
            request = {key: value for key, value in (('clinical_note', note), ('question', question)) if value}
            expected = recommendation_dur(case.emr, clinical_note=note, question=question, findings=None)
            _check_scenario(case.pid, scenario, expected)
            cases.append({'schema_version': SCHEMA_VERSION, 'case_id': case.pid, 'patient_group_id': case.pid, 'origin': 'synthetic',
                          'split': split, 'category': 'recommendation', 'scenario': scenario, 'review_status': 'draft',
                          'question_template': f'{scenario}-{split}', 'patient_id': case.pid, 'emr_payload': case.emr,
                          'request': request})
            gold.append({'schema_version': SCHEMA_VERSION, 'case_id': case.pid, 'expected_medication_alerts': expected,
                         'review_status': 'draft'})
    return cases, gold


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def main():
    ROOT.mkdir(exist_ok=True)
    cases, gold = build()
    write_jsonl(ROOT / 'inputs.jsonl', cases)
    write_jsonl(ROOT / 'gold.jsonl', gold)
    rec_cases, rec_gold = build_recommendations()
    write_jsonl(ROOT / 'recommendation_inputs.jsonl', rec_cases)
    write_jsonl(ROOT / 'recommendation_gold.jsonl', rec_gold)
    names = ('inputs.jsonl', 'gold.jsonl', 'recommendation_inputs.jsonl', 'recommendation_gold.jsonl')
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    (ROOT / 'manifest.json').write_text(json.dumps({'schema_version': SCHEMA_VERSION, 'origin': 'synthetic', 'review_status': 'draft',
        'generator': 'benchmarks/generate.py', 'seed': SEED, 'dur_snapshot': dur_version(), 'hashes': hashes},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
