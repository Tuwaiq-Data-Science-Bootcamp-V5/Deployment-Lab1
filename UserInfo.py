from pydantic import BaseModel

class UserInfo(BaseModel):
    N: int
    P: int
    K: int
    ph: float