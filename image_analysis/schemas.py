from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class MedicineItem(BaseModel):
    medicine_name: Optional[str] = Field(None, description="Tên thuốc")
    effect: Optional[str] = Field(None, description="Tác dụng")
    side_effects: Optional[str] = Field(None, description="Tác dụng phụ/lưu ý")
    interaction_with_history: Optional[str] = Field(None, description="Tương tác với tiền sử bệnh")

class LabItem(BaseModel):
    test_name: Optional[str] = Field(None, description="Tên xét nghiệm")
    value: Optional[str] = Field(None, description="Giá trị")
    unit: Optional[str] = Field(None, description="Đơn vị")
    range: Optional[str] = Field(None, description="Khoảng tham chiếu")
    evaluation: Optional[str] = Field(None, description="Đánh giá")
    explanation: Optional[str] = Field(None, description="Giải thích ý nghĩa chỉ số")

class MedicineList(BaseModel):
    document_date: datetime = Field(None, description="Ngày tháng của đơn thuốc (ISO 8601 UTC format)")
    medicines: List[MedicineItem]

class LabList(BaseModel):
    document_date: datetime = Field(None, description="Ngày tháng của kết quả xét nghiệm (ISO 8601 UTC format)")
    lab: List[LabItem]
