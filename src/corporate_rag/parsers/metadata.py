from __future__ import annotations

import re


def infer_business_metadata(text: str, source: str) -> dict[str, str]:
    lower = text.lower()
    status_match = re.search(r"статус\s*:\s*(active|archive|draft|архив(?:ный)?|черновик)", lower)
    status_value = status_match.group(1) if status_match else "active"
    status = {"архив": "archive", "архивный": "archive", "черновик": "draft"}.get(
        status_value, status_value
    )
    document_type = "policy" if "политик" in lower else "instruction"
    if "стандарт" in lower:
        document_type = "standard"
    elif "регламент" in lower:
        document_type = "procedure"
    if any(term in lower for term in ("информационн", "security", "zero trust", "парол", "vpn")):
        department = "Security"
    elif any(term in lower for term in ("servicedesk", "service desk", "ит-служб", "техническ")):
        department = "IT Operations"
    elif any(term in lower for term in ("удалённ", "удаленн", "peoplehub", "hr")):
        department = "PeopleOps"
    else:
        department = "Corporate Services"
    return {"status": status, "document_type": document_type, "department": department}
