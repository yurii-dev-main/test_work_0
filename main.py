"""
Stage 1 – Request Classification Service
=========================================
Reads raw business requests from a CSV file, sends each one to the Gemini LLM,
and validates the structured JSON response into a strict Pydantic model.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

from google import genai
from google.genai import types
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal


# ---------------------------------------------------------------------------
# 1. Pydantic Data Model
# ---------------------------------------------------------------------------

class ParsedRequest(BaseModel):
    """Structured representation of a classified business request."""

    category: Literal[
        "автоматизація",
        "інтеграція",
        "звіт/аналітика",
        "баг/підтримка",
        "питання/консультація",
        "поза скоупом",
    ] = Field(
        description=(
            "Категорія запиту. Обери одне з: "
            "'автоматизація' (автоматизація процесів), "
            "'інтеграція' (підключення систем/API), "
            "'звіт/аналітика' (формування звітів або аналіз даних), "
            "'баг/підтримка' (технічна проблема або збій), "
            "'питання/консультація' (загальне запитання без конкретного завдання), "
            "'поза скоупом' (запит не стосується IT або автоматизації)."
        )
    )

    target_department: Optional[str] = Field(
        default=None,
        description=(
            "Відділ або підрозділ, якого стосується запит (наприклад: HR, маркетинг, "
            "продажі, бухгалтерія, логістика). Встанови null, якщо відділ не вказано "
            "і не можна однозначно визначити."
        ),
    )

    priority: Literal["low", "medium", "high"] = Field(
        description=(
            "Пріоритет запиту, визначений за тоном і терміновістю: "
            "'high' – є явна терміновість, система не працює, бізнес зупинений; "
            "'medium' – є конкретне завдання, але без критичної терміновості; "
            "'low' – загальне запитання або запит без термінів."
        )
    )

    short_summary: str = Field(
        description=(
            "Одне речення (максимум 25 слів), що стисло описує суть запиту. "
            "Пиши українською мовою."
        )
    )

    requested_actions: list[str] = Field(
        default_factory=list,
        description=(
            "Список конкретних дій, які просить виконати замовник. "
            "Кожен пункт — окремий рядок. Може бути порожнім списком, якщо "
            "запит є лише запитанням без чіткого завдання."
        ),
    )

    needs_clarification: bool = Field(
        description=(
            "True, якщо запит занадто розмитий, щоб розпочати роботу без "
            "уточнювальних питань. False, якщо вимоги зрозумілі."
        )
    )


# ---------------------------------------------------------------------------
# 2. System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
Ти — старший бізнес-аналітик у команді автоматизації. Твоє завдання —
аналізувати вхідні запити від бізнес-користувачів і класифікувати їх у
структурований формат JSON.

Правила роботи:
1. Відповідай ВИКЛЮЧНО валідним JSON-об'єктом. Без пояснень, без markdown,
   без коментарів — лише чистий JSON.
2. Суворо дотримуйся переліку допустимих значень для полів `category` та `priority`.
3. Для `short_summary` використовуй одне стисле речення українською мовою.
4. Якщо запит стосується конкретного відділу (HR, маркетинг, продажі, тощо) —
   вкажи його в `target_department`, інакше залиш null.
5. `needs_clarification` встанови в true лише якщо без додаткової інформації
   неможливо зрозуміти, що саме потрібно зробити.
6. `requested_actions` — це конкретні кроки/дії, про які просить користувач.
   Якщо дій не визначено — повертай порожній список [].

JSON-схема відповіді (дотримуйся точно):
{
  "category": "<одне з допустимих значень>",
  "target_department": "<рядок або null>",
  "priority": "<low | medium | high>",
  "short_summary": "<одне речення>",
  "requested_actions": ["<дія 1>", "<дія 2>"],
  "needs_clarification": <true | false>
}
""".strip()


# ---------------------------------------------------------------------------
# 3. Data Ingestion
# ---------------------------------------------------------------------------

def load_requests(filepath: str) -> pd.DataFrame:
    """
    Load requests from a CSV file.

    Args:
        filepath: Path to the CSV file with columns: id, channel, timestamp, raw_text.

    Returns:
        A pandas DataFrame with the loaded data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the required 'raw_text' column is missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath!r}")

    df = pd.read_csv(filepath, encoding="utf-8")

    required_columns = {"id", "channel", "timestamp", "raw_text"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    print(f"[load_requests] Loaded {len(df)} row(s) from '{filepath}'.")
    return df


# ---------------------------------------------------------------------------
# 4. LLM Integration
# ---------------------------------------------------------------------------

def classify_request(raw_text: str) -> ParsedRequest:
    """
    Send a raw request text to the Gemini model and parse the response
    into a validated ParsedRequest Pydantic model.

    Args:
        raw_text: The unstructured text of the business request.

    Returns:
        A fully validated ParsedRequest instance.

    Raises:
        ValueError: If the model response cannot be parsed or validated.
        google.api_core.exceptions.GoogleAPIError: On API-level failures.
    """
    client = genai.Client()

    user_prompt = f"Класифікуй наступний запит:\n\n{raw_text}"

    print("[classify_request] Sending request to Gemini...")
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ParsedRequest,
            temperature=0.1,
            top_p=0.9,
        )
    )

    raw_json: str = response.text.strip()
    print(f"[classify_request] Raw LLM response:\n{raw_json}\n")

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned non-JSON output.\nRaw response: {raw_json!r}"
        ) from exc

    try:
        parsed = ParsedRequest.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError
        raise ValueError(
            f"LLM response failed Pydantic validation.\nData: {data}\nError: {exc}"
        ) from exc

    return parsed


# ---------------------------------------------------------------------------
# 5. Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Load environment variables from .env (GEMINI_API_KEY) ---
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "[ERROR] GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and fill in your key.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load CSV ---
    CSV_PATH = "input_requests.csv"
    df = load_requests(CSV_PATH)

    # --- Process the first row ---
    first_row = df.iloc[0]
    print(
        f"\n{'='*60}\n"
        f"Processing request #{first_row['id']} "
        f"(channel: {first_row['channel']}, "
        f"timestamp: {first_row['timestamp']})\n"
        f"Raw text:\n  {first_row['raw_text']}\n"
        f"{'='*60}\n"
    )

    result: ParsedRequest = classify_request(first_row["raw_text"])

    # --- Print validated Pydantic object ---
    print("✅ ParsedRequest (validated):")
    print(result.model_dump_json(indent=2))
