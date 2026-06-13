import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional
from xml.etree import ElementTree

from django.core.exceptions import ImproperlyConfigured
from django.db import OperationalError, ProgrammingError
from django.utils import timezone

from .models import MdFeeItem


HOSPITAL_PRICE_OPTIONS = (
    {"field": "price_unprc1", "code": "upper_general", "category": "general", "label": "상급종합병원"},
    {"field": "price_unprc2", "code": "clinic", "category": "clinic", "label": "의원·치과의원"},
    {"field": "price_unprc3", "code": "general_hospital", "category": "general", "label": "종합병원"},
    {"field": "price_unprc4", "code": "dental_hospital", "category": "general", "label": "치과병원"},
    {"field": "price_unprc5", "code": "reserved_5", "category": "general", "label": "기타 종별"},
    {"field": "price_unprc6", "code": "dental_university", "category": "dental_univ", "label": "치과대학부속병원"},
)


MDFEE_ENDPOINT = "https://apis.data.go.kr/B551182/mdfeeCrtrInfoService/getDiagnossMdfeeList"
MDFEE_SOURCE = "HIRA_MD_FEE"
SURCHARGE_TERMS = ("야간", "토요", "공휴")

LESION_FEE_MAPPINGS = {
    # 얕은 충치: 직접수복(아말감·레진·인레이) + 글라스아이오노머 + 치수복조(신경 근접 시)
    "caries": {
        "treatment_name": "치아우식증 수복치료",
        "keywords": ["아말감", "복합레진", "인레이", "글라스아이오노머", "치수복조"],
        "treatment_candidates": [
            {"name": "아말감 수복", "keywords": ["아말감"]},
            {"name": "광중합형 복합레진 충전", "keywords": ["복합레진"]},
            {"name": "인레이 수복", "keywords": ["인레이"]},
            {"name": "글라스아이오노머 수복", "keywords": ["글라스아이오노머"]},
            {"name": "치수복조", "keywords": ["치수복조"]},
        ],
    },
    "periapical_lesion": {
        "treatment_name": "치근단 및 근관 처치",
        "keywords": ["치근단절제", "발수", "근관확대", "근관충전"],
        "treatment_candidates": [
            {"name": "치근단절제술", "keywords": ["치근단절제"]},
            {"name": "발수", "keywords": ["발수"]},
            {"name": "근관확대", "keywords": ["근관확대"]},
            {"name": "근관충전", "keywords": ["근관충전"]},
        ],
    },
    "impacted_tooth": {
        "treatment_name": "매복치 발치술",
        "keywords": ["매복치", "완전매복치", "부분매복치"],
        "treatment_candidates": [
            {"name": "매복치 발치술", "keywords": ["매복치"]},
            {"name": "완전매복치 발치술", "keywords": ["완전매복치"]},
            {"name": "부분매복치 발치술", "keywords": ["부분매복치"]},
        ],
    },
    "bone_loss": {
        "treatment_name": "치주질환 기본치료",
        # HIRA korNm은 "치조골결손부골이식술"처럼 공백 없이 등록되어 있다.
        "keywords": ["치석제거", "치근활택술", "치주소파술", "치은박리소파술", "치조골결손부골이식술", "조직유도재생술"],
        "treatment_candidates": [
            {"name": "치석제거", "keywords": ["치석제거"]},
            {"name": "치근활택술", "keywords": ["치근활택술"]},
            {"name": "치주소파술", "keywords": ["치주소파술"]},
            {"name": "치은박리소파술", "keywords": ["치은박리소파술"]},
            {"name": "치조골결손부 골이식술", "keywords": ["치조골결손부골이식술"]},
            {"name": "조직유도재생술", "keywords": ["조직유도재생술"]},
        ],
    },
    "furcation_involvement": {
        "treatment_name": "치근 이개부 치주수술",
        # "치근분리술"은 HIRA 항목에 없어 분할 발치(발치술-복잡매복치[치아분할술])로 대체한다.
        "keywords": ["치은박리소파술", "치조골결손부골이식술", "조직유도재생술", "치근절제술", "치아분할"],
        "treatment_candidates": [
            {"name": "치은박리소파술", "keywords": ["치은박리소파술"]},
            {"name": "치조골결손부 골이식술", "keywords": ["치조골결손부골이식술"]},
            {"name": "조직유도재생술", "keywords": ["조직유도재생술"]},
            {"name": "치근절제술", "keywords": ["치근절제술"]},
            {"name": "치아분할술 동반 발치", "keywords": ["치아분할"]},
        ],
    },
    "retained_root": {
        "treatment_name": "잔존치근 발거",
        "keywords": ["잔존치근", "발치술", "난발치"],
        "treatment_candidates": [
            {"name": "잔존치근 발거", "keywords": ["잔존치근"]},
            {"name": "발치술", "keywords": ["발치술"]},
            {"name": "난발치", "keywords": ["난발치"]},
        ],
    },
}

BONE_LOSS_SEVERITY_TREATMENT_PROFILES = {
    "mild": {
        "treatment_name": "치주질환 초기 비수술 치료",
        "candidate_names": ["치석제거", "치근활택술"],
    },
    "medium": {
        "treatment_name": "치주질환 중등도 치료",
        "candidate_names": ["치석제거", "치근활택술", "치주소파술"],
    },
    "severe": {
        "treatment_name": "치주질환 외과적 치료",
        "candidate_names": ["치은박리소파술", "치조골결손부 골이식술", "조직유도재생술"],
    },
}

# PDCNN FI 라벨은 mild/severe 2단계만 제공된다.
# mild: 이개부 초기 침범 -> 판막 접근 소파 중심, severe: 진행성 골파괴 -> 재생/절제 수술 후보.
FURCATION_INVOLVEMENT_SEVERITY_TREATMENT_PROFILES = {
    "mild": {
        "treatment_name": "이개부 병변 초기 치주수술",
        "candidate_names": ["치은박리소파술"],
    },
    "severe": {
        "treatment_name": "이개부 병변 진행성 수술 치료",
        "candidate_names": ["치조골결손부 골이식술", "조직유도재생술", "치근절제술", "치아분할술 동반 발치"],
    },
}

PERIODONTAL_SEVERITY_TREATMENT_PROFILES = {
    "bone_loss": BONE_LOSS_SEVERITY_TREATMENT_PROFILES,
    "furcation_involvement": FURCATION_INVOLVEMENT_SEVERITY_TREATMENT_PROFILES,
}

DEFAULT_PERIAPICAL_FOLLOWUP_LABEL = "nonsurgical_endo"

PERIAPICAL_FOLLOWUP_PROFILES = {
    "nonsurgical_endo": {
        "display_label": "비외과적 근관치료 우선",
        "description": "발수, 근관확대, 근관충전 같은 비외과적 근관치료 항목을 먼저 검토합니다.",
        "next_step": "치과에서 생활력 검사와 추가 방사선 평가 후 근관치료 여부를 확인해 보세요.",
        "treatment_name": "비외과적 근관치료",
        "candidate_names": ["발수", "근관확대", "근관충전"],
    },
    "surgical_endo": {
        "display_label": "치근단절제술 고려",
        "description": "치근단절제술 같은 수술적 접근 후보를 우선 검토합니다.",
        "next_step": "기존 근관치료 여부와 병소 범위를 확인한 뒤 치근단절제술 가능성을 상담해 보세요.",
        "treatment_name": "치근단 수술적 치료",
        "candidate_names": ["치근단절제술"],
    },
    "combined_endo_surgery": {
        "display_label": "근관치료와 치근단수술 동시 검토",
        "description": "비외과적 근관치료와 치근단절제술 후보를 함께 검토합니다.",
        "next_step": "재근관치료와 수술적 접근을 함께 검토할 수 있는지 치과에서 확인해 보세요.",
        "treatment_name": "근관치료 및 치근단 수술 검토",
        "candidate_names": ["발수", "근관확대", "근관충전", "치근단절제술"],
    },
}


@dataclass
class SyncStats:
    requested: int = 0
    received: int = 0
    saved: int = 0
    skipped: int = 0
    errors: int = 0
    error_messages: Optional[List[str]] = None

    def add_error(self, keyword: str, lesion_class: str, exc: Exception) -> None:
        self.errors += 1
        if self.error_messages is None:
            self.error_messages = []
        self.error_messages.append(f"{lesion_class}/{keyword}: {exc}")


def _item_text(item: ElementTree.Element, tag: str) -> str:
    child = item.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _first_item_text(item: ElementTree.Element, *tags: str) -> str:
    for tag in tags:
        value = _item_text(item, tag)
        if value:
            return value
    return ""


def _parse_price(raw: str) -> Optional[int]:
    normalized = raw.replace(",", "").strip()
    if not normalized:
        return None
    try:
        return int(float(normalized))
    except ValueError:
        return None


def _parse_unit_price(item: ElementTree.Element) -> Optional[int]:
    direct = _parse_price(_item_text(item, "unprc"))
    if direct and direct > 0:
        return direct

    prices = []
    for tag in ("unprc1", "unprc2", "unprc3", "unprc4", "unprc5", "unprc6"):
        price = _parse_price(_item_text(item, tag))
        if price and price > 0:
            prices.append(price)
    if prices:
        return max(prices)
    return direct


def _parse_price_fields(item: ElementTree.Element) -> Dict[str, Optional[int]]:
    return {
        "price_unprc1": _parse_price(_item_text(item, "unprc1")),
        "price_unprc2": _parse_price(_item_text(item, "unprc2")),
        "price_unprc3": _parse_price(_item_text(item, "unprc3")),
        "price_unprc4": _parse_price(_item_text(item, "unprc4")),
        "price_unprc5": _parse_price(_item_text(item, "unprc5")),
        "price_unprc6": _parse_price(_item_text(item, "unprc6")),
    }


def _normalize_procedure_type(raw_value: str) -> str:
    normalized = (raw_value or "").strip()
    if normalized == "수술":
        return "surgery"
    if normalized == "비수술":
        return "non_surgery"
    return ""


def _infer_surgery_role(raw_name: str) -> str:
    normalized = (raw_name or "").strip()
    if not normalized:
        return ""
    return "secondary" if "제2의수술" in normalized else "primary"


def _has_disability_surcharge(raw_name: str) -> bool:
    normalized = (raw_name or "").strip()
    return any(token in normalized for token in ("장애인가산", "장애인 가산", "장애가산"))


def _allowed_hospital_categories(raw_name: str) -> Optional[set[str]]:
    normalized = (raw_name or "").strip()
    if not normalized:
        return None
    if any(token in normalized for token in ("종병이상", "치대부속", "상급종합", "치과대학부속병원")):
        return {"general", "dental_univ"}
    if any(token in normalized for token in ("의원·치과의원", "치과의원", "보건의료원")):
        return {"clinic"}
    return None


def _build_price_options(row: Dict[str, object], option_name: str, raw_name: str) -> List[Dict[str, object]]:
    options = []
    allowed_categories = _allowed_hospital_categories(raw_name)
    for price_meta in HOSPITAL_PRICE_OPTIONS:
        price = row.get(price_meta["field"])
        # 0원은 해당 기관 종별에 적용되지 않는 항목이므로 가격 옵션에서 제외한다.
        if price is None or price <= 0:
            continue
        if allowed_categories is not None and price_meta["category"] not in allowed_categories:
            continue
        options.append(
            {
                "name": option_name or "기본",
                "full_name": raw_name,
                "price": price,
                "pay_tp_nm": row["pay_tp_nm"] or "",
                "hospital_code": price_meta["code"],
                "hospital_category": price_meta["category"],
                "hospital_label": price_meta["label"],
                "procedure_type": row.get("procedure_type") or "",
                "surgery_role": row.get("surgery_role") or _infer_surgery_role(raw_name) or "primary",
                "disability_surcharge": bool(row.get("disability_surcharge")) or _has_disability_surcharge(raw_name),
            }
        )

    if not options and row.get("unit_price") is not None:
        options.append(
            {
                "name": option_name or "기본",
                "full_name": raw_name,
                "price": row["unit_price"],
                "pay_tp_nm": row["pay_tp_nm"] or "",
                "hospital_code": "unknown",
                "hospital_category": "all",
                "hospital_label": "기관 구분 미상",
                "procedure_type": row.get("procedure_type") or "",
                "surgery_role": row.get("surgery_role") or _infer_surgery_role(raw_name) or "primary",
                "disability_surcharge": bool(row.get("disability_surcharge")) or _has_disability_surcharge(raw_name),
            }
        )

    return options


def _parse_yyyymmdd(raw: str):
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


def parse_mdfee_items(xml_body: bytes) -> List[Dict[str, object]]:
    root = ElementTree.fromstring(xml_body)
    parsed = []
    for item in root.findall(".//item"):
        price_fields = _parse_price_fields(item)
        unit_price = _parse_unit_price(item)
        if unit_price is None:
            continue

        raw_item = {child.tag: (child.text or "").strip() for child in list(item)}
        kor_nm = _first_item_text(item, "korNm")
        parsed.append(
            {
                "mdfee_cd": _first_item_text(item, "mdfeeCd", "mdFeeCd"),
                "mdfee_div_no": _first_item_text(item, "mdfeeDivNo", "mdFeeDivNo"),
                "kor_nm": kor_nm,
                "pay_tp_nm": _first_item_text(item, "payTpNm", "payTpCd"),
                "unit_price": unit_price,
                "adtsta_dd": _parse_yyyymmdd(_first_item_text(item, "adtstaDd", "adtStaDd")),
                "cval_pnt": _item_text(item, "cvalPnt"),
                "procedure_type": _normalize_procedure_type(_first_item_text(item, "soprTpNm")),
                "surgery_role": _infer_surgery_role(kor_nm),
                "disability_surcharge": _has_disability_surcharge(kor_nm),
                "raw_item": raw_item,
                **price_fields,
            }
        )
    return parsed


def _candidate_mdfee_endpoints() -> List[str]:
    configured = os.getenv("DENTAL_MDFEE_ENDPOINT", MDFEE_ENDPOINT).strip()
    candidates = [configured]
    if configured.startswith("https://"):
        candidates.append("http://" + configured[len("https://"):])
    elif configured.startswith("http://"):
        candidates.append("https://" + configured[len("http://"):])

    deduped = []
    seen = set()
    for endpoint in candidates:
        if endpoint and endpoint not in seen:
            deduped.append(endpoint)
            seen.add(endpoint)
    return deduped


def fetch_mdfee_items(service_key: str, keyword: str, num_rows: int = 50, timeout: float = 3.0):
    params = {
        "ServiceKey": service_key,
        "pageNo": "1",
        "numOfRow": str(num_rows),
        "numOfRows": str(num_rows),
        "korNm": keyword,
    }
    last_exc = None
    for endpoint in _candidate_mdfee_endpoints():
        url = f"{endpoint}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": "dental-ai/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return parse_mdfee_items(response.read())
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return []


def iter_fee_mapping_rows() -> Iterable[Dict[str, str]]:
    for lesion_class, config in LESION_FEE_MAPPINGS.items():
        for keyword in config["keywords"]:
            yield {
                "lesion_class": lesion_class,
                "treatment_name": config["treatment_name"],
                "keyword": keyword,
            }


def _current_mapping_pairs(lesion_classes: Optional[set[str]] = None):
    return {
        (row["lesion_class"], row["keyword"])
        for row in iter_fee_mapping_rows()
        if lesion_classes is None or row["lesion_class"] in lesion_classes
    }


def _build_keyword_only_estimate(mapping: Dict[str, object]) -> Dict[str, object]:
    raw_candidates = mapping.get("treatment_candidates") or []
    candidates = raw_candidates if isinstance(raw_candidates, list) else []
    items = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        keywords = candidate.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = [str(keywords)]
        normalized_keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
        items.append(
            {
                "keyword": ", ".join(normalized_keywords),
                "lookup_keywords": normalized_keywords,
                "name": str(candidate.get("name") or mapping["treatment_name"]),
                "kor_nm": str(candidate.get("name") or mapping["treatment_name"]),
                "fee_min": None,
                "fee_max": None,
                "price": None,
                "unit_price": None,
                "copay_rate": "미산정",
                "options": [],
                "variant_count": 0,
                "adtsta_dd": None,
                "lookup_status": "keyword_only",
            }
        )

    return {
        "treatment_name": mapping["treatment_name"],
        "fee_min": None,
        "fee_max": None,
        "currency": "KRW",
        "source": MDFEE_SOURCE,
        "items": items,
        "treatment_items": items,
        "lookup_keywords": list(mapping.get("keywords") or []),
        "lookup_status": "keyword_only",
    }


def _build_mapping_from_candidate_names(
    base_mapping: Dict[str, object],
    treatment_name: str,
    candidate_names: List[str],
) -> Dict[str, object]:
    raw_candidates = base_mapping.get("treatment_candidates") or []
    candidates = [
        candidate
        for candidate in raw_candidates
        if isinstance(candidate, dict) and str(candidate.get("name") or "") in candidate_names
    ]
    keywords: List[str] = []
    for candidate in candidates:
        for keyword in candidate.get("keywords") or []:
            normalized = str(keyword).strip()
            if normalized and normalized not in keywords:
                keywords.append(normalized)
    return {
        **base_mapping,
        "treatment_name": treatment_name,
        "keywords": keywords,
        "treatment_candidates": candidates,
    }


def _resolve_treatment_mapping(
    lesion_class: str,
    severity_class_name: Optional[str] = None,
    followup_class_name: Optional[str] = None,
    followup_applied: bool = False,
) -> Optional[Dict[str, object]]:
    base_mapping = LESION_FEE_MAPPINGS.get(lesion_class)
    if not base_mapping:
        return None

    if lesion_class == "periapical_lesion":
        profile_name = (
            str(followup_class_name or "").strip()
            if followup_applied and str(followup_class_name or "").strip() in PERIAPICAL_FOLLOWUP_PROFILES
            else DEFAULT_PERIAPICAL_FOLLOWUP_LABEL
        )
        profile = PERIAPICAL_FOLLOWUP_PROFILES.get(profile_name)
        if profile:
            return _build_mapping_from_candidate_names(
                base_mapping,
                str(profile["treatment_name"]),
                [str(name) for name in profile["candidate_names"]],
            )
        return base_mapping

    severity_profiles = PERIODONTAL_SEVERITY_TREATMENT_PROFILES.get(lesion_class)
    if not severity_profiles:
        return base_mapping

    severity_key = str(severity_class_name or "").strip().lower()
    profile = severity_profiles.get(severity_key)
    if not profile:
        return base_mapping

    return _build_mapping_from_candidate_names(
        base_mapping,
        str(profile["treatment_name"]),
        [str(name) for name in profile["candidate_names"]],
    )


def _resolve_periapical_followup_route(detection: Dict[str, object]) -> Optional[Dict[str, object]]:
    if str(detection.get("class_name", "")) != "periapical_lesion":
        return None

    model_label = str(detection.get("followup_class_name", "")).strip()
    model_applied = bool(detection.get("followup_applied"))
    if model_applied and model_label in PERIAPICAL_FOLLOWUP_PROFILES:
        route_name = model_label
        route_source = "model"
    else:
        route_name = DEFAULT_PERIAPICAL_FOLLOWUP_LABEL
        route_source = "default"

    profile = PERIAPICAL_FOLLOWUP_PROFILES.get(route_name)
    if not profile:
        return None

    return {
        "route_name": route_name,
        "display_label": str(profile["display_label"]),
        "description": str(profile["description"]),
        "next_step": str(profile["next_step"]),
        "source": route_source,
        "confidence": detection.get("followup_confidence") if route_source == "model" else None,
    }


def _split_top_level_fee_name(raw_name: str):
    square_depth = 0
    paren_depth = 0

    for index, char in enumerate(raw_name):
        if char == "[":
            square_depth += 1
            continue
        if char == "]":
            square_depth = max(0, square_depth - 1)
            continue
        if char == "(":
            paren_depth += 1
            continue
        if char == ")":
            paren_depth = max(0, paren_depth - 1)
            continue
        if square_depth == 0 and paren_depth == 0 and char in "-/":
            return raw_name[:index], raw_name[index + 1 :], char

    return raw_name, "", ""


def _parse_fee_display_names(raw_name: str):
    normalized = (raw_name or "").strip().rstrip("/")
    if not normalized:
        return "", "기본"

    name_part, option_part, delimiter = _split_top_level_fee_name(normalized)
    display_name = name_part.strip() or normalized
    option_name = option_part.strip(" /")

    unit_match = re.match(r"^(.*?)(\[[^\[\]]+\])$", display_name)
    if unit_match:
        display_name = unit_match.group(1).strip() or display_name
        unit_label = unit_match.group(2).strip("[]")
        option_name = f"{unit_label} / {option_name}" if option_name else unit_label

    if not option_name and delimiter == "/":
        option_name = "기본"

    return display_name or normalized, option_name or "기본"


def sync_mdfee_items(
    service_key: Optional[str] = None,
    timeout: Optional[float] = None,
    lesion_classes: Optional[Iterable[str]] = None,
) -> SyncStats:
    service_key = service_key or os.getenv("DATAGO_KEY")
    if not service_key:
        raise ImproperlyConfigured("DATAGO_KEY is not set.")

    if timeout is None:
        timeout = float(os.getenv("DATAGO_MDFEE_TIMEOUT", "30"))

    lesion_class_set = {str(name).strip() for name in (lesion_classes or []) if str(name).strip()} or None
    stats = SyncStats()
    synced_at = timezone.now()
    current_pairs = _current_mapping_pairs(lesion_class_set)
    existing_rows = MdFeeItem.objects.all().only("id", "lesion_class", "keyword")
    if lesion_class_set is not None:
        existing_rows = existing_rows.filter(lesion_class__in=lesion_class_set)
    for item in existing_rows:
        if (item.lesion_class, item.keyword) not in current_pairs:
            item.delete()

    for mapping in iter_fee_mapping_rows():
        if lesion_class_set is not None and mapping["lesion_class"] not in lesion_class_set:
            continue
        stats.requested += 1
        try:
            items = fetch_mdfee_items(service_key, mapping["keyword"], timeout=timeout)
        except Exception as exc:
            stats.add_error(mapping["keyword"], mapping["lesion_class"], exc)
            continue
        stats.received += len(items)

        for item in items:
            if not item["mdfee_cd"] or item["adtsta_dd"] is None:
                stats.skipped += 1
                continue

            MdFeeItem.objects.update_or_create(
                mdfee_cd=item["mdfee_cd"],
                adtsta_dd=item["adtsta_dd"],
                keyword=mapping["keyword"],
                lesion_class=mapping["lesion_class"],
                defaults={
                    "treatment_name": mapping["treatment_name"],
                    "mdfee_div_no": item["mdfee_div_no"],
                    "kor_nm": item["kor_nm"],
                    "pay_tp_nm": item["pay_tp_nm"],
                    "unit_price": item["unit_price"],
                    "price_unprc1": item["price_unprc1"],
                    "price_unprc2": item["price_unprc2"],
                    "price_unprc3": item["price_unprc3"],
                    "price_unprc4": item["price_unprc4"],
                    "price_unprc5": item["price_unprc5"],
                    "price_unprc6": item["price_unprc6"],
                    "cval_pnt": item["cval_pnt"],
                    "procedure_type": item["procedure_type"],
                    "surgery_role": item["surgery_role"],
                    "disability_surcharge": item["disability_surcharge"],
                    "raw_item": item["raw_item"],
                    "synced_at": synced_at,
                },
            )
            stats.saved += 1
    return stats


def build_treatment_estimate(
    lesion_class: str,
    severity_class_name: Optional[str] = None,
    followup_class_name: Optional[str] = None,
    followup_applied: bool = False,
) -> Optional[Dict[str, object]]:
    mapping = _resolve_treatment_mapping(
        lesion_class,
        severity_class_name,
        followup_class_name,
        followup_applied,
    )
    if not mapping:
        return None

    try:
        rows = list(
            MdFeeItem.objects.filter(lesion_class=lesion_class)
            .order_by("keyword", "-adtsta_dd", "unit_price")
            .values(
                "keyword",
                "treatment_name",
                "mdfee_cd",
                "mdfee_div_no",
                "kor_nm",
                "pay_tp_nm",
                "unit_price",
                "price_unprc1",
                "price_unprc2",
                "price_unprc3",
                "price_unprc4",
                "price_unprc5",
                "price_unprc6",
                "procedure_type",
                "surgery_role",
                "disability_surcharge",
                "adtsta_dd",
            )
        )
    except (OperationalError, ProgrammingError):
        rows = []

    if not rows:
        return _build_keyword_only_estimate(mapping)

    rows = [
        row
        for row in rows
        if not any(term in (row["kor_nm"] or "") for term in SURCHARGE_TERMS)
    ]
    mapping_keywords = {str(keyword).strip() for keyword in mapping.get("keywords") or [] if str(keyword).strip()}
    if mapping_keywords:
        rows = [row for row in rows if str(row.get("keyword") or "").strip() in mapping_keywords]
    if not rows:
        return _build_keyword_only_estimate(mapping)

    grouped = {}
    all_prices = []

    for row in rows:
        raw_name = row["kor_nm"] or ""
        name, option_name = _parse_fee_display_names(raw_name)
        grouped.setdefault(
            name,
            {
                "name": name,
                "prices": [],
                "pay_types": set(),
                "keywords": set(),
                "options": [],
                "option_keys": set(),
                "latest_adtsta_dd": None,
                "variant_count": 0,
            },
        )
        group = grouped[name]
        group["pay_types"].add(row["pay_tp_nm"] or "")
        group["keywords"].add(row["keyword"] or "")

        for option in _build_price_options(row, option_name, raw_name):
            option_key = (
                row["mdfee_cd"],
                row["mdfee_div_no"],
                raw_name,
                option["hospital_code"],
                option["price"],
                option["surgery_role"],
                option["disability_surcharge"],
                row["pay_tp_nm"] or "",
            )
            if option_key in group["option_keys"]:
                continue
            group["option_keys"].add(option_key)
            group["variant_count"] += 1
            group["prices"].append(option["price"])
            group["options"].append(option)
            all_prices.append(option["price"])

        if row["adtsta_dd"] and (group["latest_adtsta_dd"] is None or row["adtsta_dd"] > group["latest_adtsta_dd"]):
            group["latest_adtsta_dd"] = row["adtsta_dd"]

    if not all_prices:
        return _build_keyword_only_estimate(mapping)

    items = []
    for group in sorted(grouped.values(), key=lambda item: (min(item["prices"]), item["name"]))[:12]:
        item_prices = [price for price in group["prices"] if price is not None]
        if not item_prices:
            continue
        pay_types = {pay_type for pay_type in group["pay_types"] if pay_type}
        if any("비급여" in pay_type for pay_type in pay_types):
            copay_rate = "100%"
        elif any("급여" in pay_type for pay_type in pay_types):
            copay_rate = "급여 기준"
        else:
            copay_rate = "미산정"
        adtsta_dd = group["latest_adtsta_dd"]
        items.append(
            {
                "keyword": ", ".join(sorted(group["keywords"])),
                "name": group["name"],
                "kor_nm": group["name"],
                "fee_min": min(item_prices),
                "fee_max": max(item_prices),
                "price": min(item_prices),
                "unit_price": min(item_prices),
                "copay_rate": copay_rate,
                "options": sorted(
                    group["options"],
                    key=lambda option: (
                        option["price"],
                        option.get("hospital_category") or "",
                        option.get("hospital_label") or "",
                        option["name"],
                    ),
                ),
                "variant_count": group["variant_count"],
                "adtsta_dd": adtsta_dd.strftime("%Y%m%d") if adtsta_dd else None,
                "lookup_keywords": sorted(group["keywords"]),
                "lookup_status": "priced",
            }
        )

    return {
        "treatment_name": mapping["treatment_name"],
        "fee_min": min(all_prices),
        "fee_max": max(all_prices),
        "currency": "KRW",
        "source": MDFEE_SOURCE,
        "items": items,
        "treatment_items": items,
        "lookup_keywords": list(mapping.get("keywords") or []),
        "lookup_status": "priced",
        "severity_basis": severity_class_name,
        "followup_basis": followup_class_name,
    }


def attach_treatment_estimates(result: Dict[str, object]) -> Dict[str, object]:
    detections = result.get("detections")
    if not isinstance(detections, list):
        result["fee_estimate_enabled"] = False
        result["fee_source"] = MDFEE_SOURCE
        result["fee_error"] = "No detections list."
        return result

    any_estimate = False
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        periapical_route = _resolve_periapical_followup_route(detection)
        if periapical_route:
            detection["followup_display_label"] = periapical_route["display_label"]
            detection["followup_description"] = periapical_route["description"]
            detection["followup_next_step"] = periapical_route["next_step"]
            detection["followup_source"] = periapical_route["source"]
            if not detection.get("followup_class_name"):
                detection["followup_class_name"] = periapical_route["route_name"]
        lesion_class = str(detection.get("class_name", ""))
        severity_basis = str(detection.get("severity_class_name", "")) or None
        estimate = build_treatment_estimate(
            lesion_class,
            severity_basis,
            str(detection.get("followup_class_name", "")) or None,
            bool(detection.get("followup_applied")),
        )
        if estimate:
            if periapical_route:
                estimate["followup_display_label"] = periapical_route["display_label"]
                estimate["followup_description"] = periapical_route["description"]
                estimate["followup_next_step"] = periapical_route["next_step"]
                estimate["followup_source"] = periapical_route["source"]
            detection["treatment_estimate"] = estimate
            any_estimate = True

    result["fee_estimate_enabled"] = any_estimate
    result["fee_source"] = MDFEE_SOURCE if any_estimate else None
    result["fee_error"] = None if any_estimate else "No stored HIRA fee items for detected lesions."
    return result
