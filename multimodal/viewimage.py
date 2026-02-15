import base64
import openai
from dotenv import load_dotenv

class ViewImage:
    def __init__(self):
        load_dotenv()

        self.model_alias = "qwen3-vl:8b-instruct"
        self.endpoint = "http://localhost:11434/v1"
        self.llm = openai.OpenAI(
            base_url = self.endpoint
        )

    def encode_data_to_base64(self, data_path) -> str:
        with open(data_path, "rb") as f:
            all_bytes = f.read()
        return base64.b64encode(all_bytes).decode('utf-8')
    
    def get_image_description(self, image_path: str) -> str:
        image_base64 = self.encode_data_to_base64(image_path)
        data_url = f"data:{"image/jpeg"};base64,{image_base64}"
        prompt = [
        {"role": "system", "content": "You are an image description generator."},
        {"role": "user", "content": [
            {
                "type": "text", "text": "Please describe the image"
            },
            {
                "type": "image_url", "image_url": {"url": data_url}
            }
        ]}
        ]
        response = self.llm.chat.completions.create(
            model=self.model_alias,
            messages=prompt,
            stream=False
        )
        return response.choices[0].message.content
    
if __name__ == "__main__":
    agent = ViewImage()
    image_path = "./media/fall.jpg"
    description = agent.get_image_description(image_path)
    print("Description of the image:", description)