"""
We will build a Travel Recommendation System that provides personalized travel suggestions based on user 
preferences and current location. The system will utilize a Context Manager to manage user preferences and 
current context, allowing it to generate tailored travel recommendations.
"""
from dataclasses import dataclass

@dataclass
class UserPreference:
    user_id: int
    name: str
    favorite_sports: str
    favorite_food: str

@dataclass
class UserContext:
    user_id: int
    current_location: str

class DummyUserPerferenceManager:
    def __init__(self):
        self.user_profs = [
            UserPreference(user_id=1, name="Alice", favorite_sports="Tennis", favorite_food="Pizza"),
            UserPreference(user_id=2, name="Bob", favorite_sports="Football", favorite_food="Sushi"),
            UserPreference(user_id=3, name="Charlie", favorite_sports="Basketball", favorite_food="Burgers"),
            UserPreference(user_id=4, name="Diana", favorite_sports="Swimming", favorite_food="Salad"),
            UserPreference(user_id=5, name="Eve", favorite_sports="Rugby", favorite_food="Pasta")
    ]

    def get_user_preference(self, user_id: int):
        up = next((prof for prof in self.user_profs if prof.user_id == user_id), None)
        if up:
            return f"User {up.name} prefers {up.favorite_sports} in sports, and in food prefers {up.favorite_food}."
        else:
            return "User not found."

class ContextManager:
    def __init__(self) -> None:
        pass

if __name__ == "__main__":
    du = DummyUserPerferenceManager()
    print(du.get_user_preference(2))