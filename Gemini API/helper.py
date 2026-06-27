"""
IMPORTS & API KEY
"""
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

"""
HELPER FUNCTION
"""
def get_response(chat_context, model='gemini-2.5-flash', temperature=1.0, max_tokens=500):
    system_instruction = None
    formatted_contents = []
    
    for turn in chat_context:
        if turn['role'] == 'system':
            system_instruction = turn['content']
        else:
            role = 'model' if turn['role'] == 'assistant' else 'user'
            formatted_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=turn['content'])]
                )
            )

    api_response = client.models.generate_content(
        model=model,
        contents=formatted_contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction
        )
    )

    return api_response.text

"""
EXAMPLE
"""
chat_context =  [  
    {
        'role': 'system', 
        'content': 'You are an assistant who responds in the style of Dr Seuss.'
    },    
    {
        'role': 'user', 
        'content': 'write me a very short poem about a happy carrot'
    },  
] 

response = get_response(chat_context, temperature=2.0)
print(response)