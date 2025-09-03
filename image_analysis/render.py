def render_prescription(medicine_items):
    result = ["🔹 Thông tin thuốc trong đơn:"]
    for idx, med in enumerate(medicine_items, 1):
        med_str = f"{idx}. {med.medicine_name or '[Tên thuốc không xác định]'}  "  # 2 space
        if med.effect:
            med_str += f"\n💊 Tác dụng: {med.effect}  "
        if med.side_effects:
            med_str += f"\n⚠️ Tác dụng phụ/lưu ý: {med.side_effects}  "
        if med.interaction_with_history:
            med_str += f"\n🔄 Tương tác với tiền sử bệnh: {med.interaction_with_history}  "
        result.append(med_str)
    if len(result) == 1:
        result.append("Không có thông tin thuốc trong đơn.")
    return "\n".join(result)

def render_lab(lab_items):
    result = ["🧪 Thông tin kết quả xét nghiệm:"]
    for idx, item in enumerate(lab_items, 1):
        line = f"{idx}. {item.test_name or '[Tên xét nghiệm không xác định]'}, Chỉ số đo được: {item.value or 'Chưa rõ'} {item.unit or ''}"
        if item.range and item.range != "Chưa rõ":
            line += f" (Khoảng tham chiếu: {item.range})"
        if item.evaluation and item.evaluation != "Chưa rõ":
            line += f" — Đánh giá: {item.evaluation}  "
        result.append(line)
        if item.explanation:
            result.append(f"💡 Giải thích: {item.explanation}")
    if len(result) == 1:
        result.append("Không có dữ liệu xét nghiệm.")
    return "\n".join(result)