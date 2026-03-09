from pydantic import BaseModel

class WarmupRequest(BaseModel):
    config_path: str
    params_path: str

from typing import List, Any

class PredictRequest(BaseModel):
    values: List[Any]  # 接受任何类型的列表，更具鲁棒性
