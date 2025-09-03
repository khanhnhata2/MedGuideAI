from typing import List
from image_analysis.schemas import LabList, MedicineList
import json
import re


def render_latest_result(lab_lists: List[LabList]) -> str:

    # Sắp xếp theo ngày giảm dần (không lấy phần tử đầu tiên nữa)
    sorted_labs = sorted(lab_lists, key=lambda x: x["document_date"], reverse=True)

    result_lines = []
    for lab in sorted_labs:
        result_lines.append(f"📅 Ngày xét nghiệm: {lab['document_date'].strftime('%Y-%m-%d %H:%M:%S')}  ")
        result_lines.append("\n🧪 Kết quả xét nghiệm:")

        for idx, item in enumerate(lab["results"], start=1):
            line = f"{idx}. {item.get('test_name') or '[Tên xét nghiệm không xác định]'}  "
            if item.get("value"):
                line += f" = {item['value']}"
                if item.get("unit"):
                    line += f" {item['unit']}"
            if item.get("range"):
                line += f"(Khoảng tham chiếu: {item['range']})  "
            if item.get("evaluation"):
                line += f"\nĐánh giá: {item['evaluation']}"
            if item.get("explanation"):
                line += f"\n {item['explanation']}  "
            result_lines.append(line)

        result_lines.append("")  # Dòng trống giữa các lần xét nghiệm

    return "\n".join(result_lines).strip()

def render_lab_comparison(json_str: str) -> str:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", json_str.strip(), flags=re.DOTALL)
    result = json.loads(cleaned)
    lines = []

    # --- Tiêu đề ---
    lines.append("📊 **So sánh kết quả xét nghiệm**  \n")

    # --- Tóm tắt ---
    if "summary" in result and isinstance(result["summary"], dict):
        summary = result["summary"]
        date1 = summary.get("test_date_1", "?")
        date2 = summary.get("test_date_2", "?")
        overall = summary.get("overall_health", "")
        lines.append(f"**Lần 1:** {date1} | **Lần 2:** {date2}")
        lines.append(f"\n**Tổng quan:** {overall}")

    # --- Bảng so sánh ---
    if "table" in result and isinstance(result["table"], list):
        lines.append("\n**Chi tiết so sánh:**")
        for idx, row in enumerate(result["table"], 1):
            name = row.get("test_name", "[Không xác định]")
            v1 = row.get("value_1", "?")
            v2 = row.get("value_2", "?")
            change = row.get("change", "?")
            status = row.get("evaluation", "")
            lines.append(f"{idx}. {name}: {v1} ➡️ {v2} ({change}) - {status}")

    # --- Khuyến nghị ---
    if "recommendations" in result and isinstance(result["recommendations"], dict):
        lines.append("\n💡 **Khuyến nghị:**")
        for _, rec in result["recommendations"].items():
            lines.append(f"- {rec}")

    return "\n".join(lines)

def render_latest_prescription(medicine_lists: list) -> str:
    if not medicine_lists:
        return "❌ Không tìm thấy thông tin thuốc."

    # Lấy bản ghi mới nhất (ở đây ví dụ lấy phần tử đầu tiên)
    latest = medicine_lists[0]

    if not latest.get("medicines"):
        return "❌ Không tìm thấy thông tin thuốc."

    lines = []
    if latest.get("document_date"):
        lines.append(f"Ngày kê đơn: {latest['document_date'].strftime('%d/%m/%Y')}")

    lines.append("\nThông tin thuốc trong đơn:  ")

    for idx, med in enumerate(latest["medicines"], start=1):
        med_lines = [f"{idx}. {med.get('medicine_name') or '[Tên thuốc không xác định]'}"]

        if med.get("effect"):
            med_lines.append(f"   \n• Tác dụng: {med['effect']}")
        if med.get("side_effects"):
            med_lines.append(f"   \n• Tác dụng phụ/Lưu ý: {med['side_effects']}  ")
        if med.get("interaction_with_history"):
            med_lines.append(f"   \n• Tương tác với tiền sử bệnh: {med['interaction_with_history']}  \n")

        lines.append("\n".join(med_lines))

    return "\n".join(lines)




