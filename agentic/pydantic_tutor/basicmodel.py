from pydantic import BaseModel
from pydantic import ValidationError
from typing import Optional
from datetime import date
import json

class User(BaseModel):
    user_id: int
    first_name: str
    last_name: str
    email: str
    date_of_birth: Optional[date] = None
    salary: float

class BasicModel:
    def __init__(self):
        self.user = None

    def populate(self, data: dict):
        try:
            self.user = User(**data)
        except ValidationError as e:
            print("Validation error:", e)

    def test_valid_data(self):
        user_data = {
            "user_id": 1,
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@example.com",
            "date_of_birth": "1985-05-14",
            "salary": 50000.00
        }
        self.populate(user_data)
        return self.user
    
    def test_invalid_data(self):
        user_data = {
            "user_id": "U001",  # Invalid type, should be int
            "first_name": "Jane",
            "last_name": "Doe",
            "email": "john.doe@example.com",
            "date_of_birth": "1985-05-14"
        }
        self.populate(user_data)
        return self.user
    
    def skip_optional(self):
        user_data = {
            "user_id": 2,
            "first_name": "Alice",
            "last_name": "Smith",
            "email": "alice.smith@example.com",
            "salary": 60000.00
        }
        self.populate(user_data)
        return self.user

if __name__ == "__main__":
    model = BasicModel()
    #model.test_valid_data()
    #model.test_invalid_data()
    model.skip_optional()
    print(model.user)
