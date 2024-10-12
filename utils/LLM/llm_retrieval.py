import json
import openai

class QueryProcessor:
    def __init__(self, api_key_path):
        self.api_key = self.load_api_key(api_key_path)
        self.client = openai.Client(api_key=self.api_key)

    def load_api_key(self, api_key_path):
        try:
            with open(api_key_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            raise Exception(f"API key file not found at {api_key_path}")
        except Exception as e:
            raise Exception(f"Error reading API key: {e}")

    def get_prompt(self):
        # Generate a system prompt based on expected query types and methods
        frames = [
            {
                "query": "Tìm văn bản trên một ghi chú viết tay về chi tiết cuộc họp.",
                "method": ["openclip", "evalip", "ocr"],
            },
            {
                "query": "Tìm kiếm một bản ghi âm cuộc phỏng vấn với các điểm chính.",
                "method": ["asr"],
            }
        ]
        return str(frames)

    def get_query_variants(self, query):
        # Template for the prompt sent to the model
        temp_prompt = '''
        Generate 5 semantically similar queries by rephrasing or using synonyms. 
        For each query, assess the content type and determine the appropriate search method(s) 
        from [OCR, ASR, openclip, Caption]:

        - **OCR**: Use if the query involves searching for visible text within images.
        - **ASR**: Use if the query involves searching audio or video content for spoken words.
        - **openclip **: Use if the query focuses on visual elements or characteristics within an image.
        - **Caption**: Use if the query relates to metadata or descriptive text associated with the image.

        Ensure the output format follows this example:'''
        
        system_prompt = self.get_prompt()

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": temp_prompt + system_prompt},
                {"role": "user", "content": query}
            ]
        )

        raw_output = completion.choices[0].message.content

        cleaned_output = raw_output.replace('"', '\\"').replace("```json", "").replace("```", "")
        cleaned_output = cleaned_output.replace("'", '"')

        try:
            data = json.loads(cleaned_output)
            return data
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print("Problematic Output:\n", cleaned_output)
            return cleaned_output


if __name__ == "__main__":
    api_key_path = "api_key.txt"
    query = "Tìm văn bản trên một ghi chú viết tay về chi tiết cuộc họp."
    processor = QueryProcessor(api_key_path)
    query_variants = processor.get_query_variants(query)
    print(query_variants)
    
