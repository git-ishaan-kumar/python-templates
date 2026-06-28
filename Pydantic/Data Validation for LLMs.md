# Gemini & Pydantic Data Validation

Documentation and boilerplate for enforcing structured outputs using the Gemini API and Pydantic, including automatic retry logic for validation errors.

## Installation

```bash
pip install pydantic google-genai python-dotenv
```

## Setup & API Key

```python
import os
from google import genai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
```

## Basic Pydantic Models

A Pydantic model defines the exact structure and constraints for the data you want the LLM to return.

```python
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Literal

class UserInput(BaseModel):
    name: str
    email: EmailStr
    age: int = Field(ge=0, description="User's age. Must be positive.")
    role: Literal["admin", "user", "guest"]
    phone: Optional[str] = None
```

## Validation & Retry Helper

When generating structured data, the LLM might occasionally violate the schema. This helper catches the `ValidationError`, feeds the exact error back into the prompt, and retries the request `max_retries` times.

```python
from typing import Type, TypeVar, Optional
from pydantic import BaseModel, ValidationError
from google import genai
from google.genai import types
from google.genai.errors import APIError

T = TypeVar('T', bound=BaseModel)

def generate_structured_data(
    client: genai.Client,
    prompt: str, 
    schema: Type[T], 
    max_retries: int = 3, 
    model: str = 'gemini-3.1-flash-lite',
    temperature: float = 0.1
) -> Optional[T]:
    """Forces structured JSON output and automatically retries on validation failure."""
    current_prompt = prompt
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=current_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature
                )
            )
            return response.parsed
            
        except ValidationError as e:
            print(f"Validation Error (Attempt {attempt + 1}/{max_retries}). Retrying...")
            current_prompt += f"\n\nVALIDATION ERROR ON PREVIOUS ATTEMPT:\n{e}\nPlease fix and retry."
            
        except APIError as e:
            print(f"API Error: {e}")
            break
            
    return None
```

## Usage Examples

### Example 1: Extracting a Profile

```python
class UserProfile(BaseModel):
    name: str
    age: int
    tags: list[str]

profile = generate_structured_data(
    client=client,
    prompt="Extract this user: John Doe is 25 and loves hiking and coding.",
    schema=UserProfile,
    max_retries=3
)

if profile:
    print(profile.name)
    print(profile.tags)
```

### Example 2: Email Classifier

```python
class EmailClassification(BaseModel):
    category: Literal["work", "urgent", "personal", "spam"]
    summary: str = Field(min_length=10, max_length=200)
    requires_action: bool

email_text = "URGENT: Server downtime detected. Immediate action required."

classification = generate_structured_data(
    client=client,
    prompt=f"Classify this email: {email_text}",
    schema=EmailClassification,
    max_retries=2
)

if classification:
    print(f"Category: {classification.category}")
    print(f"Action needed: {classification.requires_action}")
```