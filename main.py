"""
Request Classification Service
==============================
Reads raw business requests from a CSV file, sends each one to the Gemini LLM asynchronously,
and validates the structured JSON response into a strict Pydantic model with error handling.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Optional, Literal
import logging

from google import genai
from google.genai import types
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# 1. Pydantic Data Model
# ---------------------------------------------------------------------------

class ParsedRequest(BaseModel):
    """Structured representation of a classified business request."""
    
    request_id: str = Field(
        description="ID запиту (наприклад, REQ-001), щоб зв'язати відповідь із вхідним текстом."
    )

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

class ParsedRequestBatch(BaseModel):
    """Колекція проаналізованих запитів. Gemini повертатиме об'єкт цього типу."""
    results: list[ParsedRequest]


# ---------------------------------------------------------------------------
# 2. System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
Ти — старший бізнес-аналітик у команді автоматизації. Твоє завдання —
аналізувати вхідні запити від бізнес-користувачів і класифікувати їх у
структурований формат JSON.

На вхід ти отримаєш один або декілька запитів. Кожен запит має свій ID та Текст.
Ти повинен обробити КОЖЕН запит та повернути масив результатів у полі `results`.

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
  "results": [
    {
      "request_id": "<ID запиту, який був переданий>",
      "category": "<одне з допустимих значень>",
      "target_department": "<рядок або null>",
      "priority": "<low | medium | high>",
      "short_summary": "<одне речення>",
      "requested_actions": ["<дія 1>", "<дія 2>"],
      "needs_clarification": <true | false>
    }
  ]
}
""".strip()


# ---------------------------------------------------------------------------
# 3. Data Ingestion
# ---------------------------------------------------------------------------

def load_requests(filepath: str) -> pd.DataFrame:
    """
    Load requests from a CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath!r}")

    df = pd.read_csv(filepath, encoding="utf-8")

    required_columns = {"id", "channel", "timestamp", "raw_text"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    logging.info(f"Loaded {len(df)} row(s) from '{filepath}'.")
    return df


# ---------------------------------------------------------------------------
# 4. LLM Integration & Error Handling
# ---------------------------------------------------------------------------

def generate_fallback(request_id: str) -> dict:
    """Helper to generate a fallback dict for a single failed request."""
    return {
        "request_id": request_id,
        "category": "поза скоупом",
        "target_department": None,
        "priority": "low",
        "short_summary": "Помилка класифікації / Не вдалося обробити запит",
        "requested_actions": [],
        "needs_clarification": True,
        "_error": True
    }

async def classify_batch(requests: list[dict], batch_delay: int, max_retries: int = 3) -> list[dict]:
    """
    Відправляє групу запитів у Gemini для класифікації.
    Має вбудований механізм повторних спроб (Retry), якщо досягнуто лімітів API (429).
    
    Args:
        requests: список словників з ключами 'id' та 'raw_text'.
        batch_delay: час очікування (в секундах) перед повторною спробою.
    """
    client = genai.Client()
    
    # Формуємо єдиний промпт для всіх запитів у цій групі
    user_prompt_parts = ["Класифікуй наступні запити:\n"]
    for req in requests:
        user_prompt_parts.append(f"--- ID: {req['id']} ---\nТекст: {req['raw_text']}\n")
    user_prompt = "\n".join(user_prompt_parts)
    
    batch_ids = [str(r["id"]) for r in requests]
    batch_label = f"[{batch_ids[0]} ... {batch_ids[-1]}]" if len(batch_ids) > 1 else f"[{batch_ids[0]}]"

    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"{batch_label} Sending request to Gemini (Attempt {attempt}/{max_retries})...")
            
            response = await client.aio.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=ParsedRequestBatch,
                    temperature=0.1,
                    top_p=0.9,
                )
            )

            raw_json = response.text.strip()
            data = json.loads(raw_json)
            parsed_batch = ParsedRequestBatch.model_validate(data)
            
            logging.info(f"{batch_label} Successfully parsed {len(parsed_batch.results)} requests.")
            return [req.model_dump() for req in parsed_batch.results]
            
        except (json.JSONDecodeError, ValidationError) as exc:
            logging.warning(f"{batch_label} Validation/Parsing failed on attempt {attempt}: {exc}")
            # Для помилок валідації не чекаємо довго, фолбечимо або можна зробити retry. 
            # Для безпеки виходимо і повертаємо fallback.
            break
            
        except Exception as exc:
            logging.warning(f"{batch_label} LLM call failed on attempt {attempt}: {exc}")
            # Перевіряємо чи це помилка лімітів (429)
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                if attempt < max_retries:
                    logging.info(f"{batch_label} Rate limit hit. Waiting {batch_delay} seconds before retry...")
                    await asyncio.sleep(batch_delay)
                    continue
            
            # Якщо це інша помилка або вичерпано спроби - виходимо з циклу
            break

    # Якщо ми вийшли з циклу, значить всі спроби вичерпано або сталася фатальна помилка
    logging.error(f"{batch_label} All attempts failed. Applying fallback.")
    return [generate_fallback(req_id) for req_id in batch_ids]


# ---------------------------------------------------------------------------
# 5. Batch Processing
# ---------------------------------------------------------------------------

async def process_all_requests(
    df: pd.DataFrame, 
    mode: Literal["economy", "heavy"] = "economy",
    rows_per_call: int = 5,
    api_calls_per_batch: int = 12, 
    batch_delay: int = 75
) -> list[dict]:
    """
    Обробляє всі запити, використовуючи один із двох режимів:
    
    1. mode="economy" (Економія запитів):
       Об'єднує декілька рядків (rows_per_call) в один запит до LLM. 
       Оптимально для збереження лімітів, коли обсяг тексту в кожному рядку невеликий.
       
    2. mode="heavy" (Перевантажені рядки):
       Відправляє кожен рядок окремим запитом до LLM (rows_per_call ігнорується).
       Оптимально для випадків, коли кожен рядок містить гігантську кількість даних, 
       і не можна розмивати контекст об'єднанням.
    """
    results = []
    
    # Визначаємо реальний розмір групи рядків для одного API-виклику
    group_size = rows_per_call if mode == "economy" else 1
    
    # Формуємо список груп (кожна група піде в 1 API-виклик)
    all_groups = []
    for i in range(0, len(df), group_size):
        group_df = df.iloc[i:i+group_size]
        # Конвертуємо DataFrame у список словників для зручності
        group_requests = group_df[["id", "raw_text"]].to_dict(orient="records")
        all_groups.append(group_requests)
        
    logging.info(f"Mode: '{mode}'. Formed {len(all_groups)} API calls (grouped by {group_size} rows).")
    
    # Відправляємо групи батчами по api_calls_per_batch
    for i in range(0, len(all_groups), api_calls_per_batch):
        current_batch_groups = all_groups[i:i+api_calls_per_batch]
        logging.info(f"\n--- Processing batch {i // api_calls_per_batch + 1} ({len(current_batch_groups)} API calls) ---")
        
        tasks = []
        for group in current_batch_groups:
            task = asyncio.create_task(classify_batch(group, batch_delay))
            tasks.append(task)
            
        # Чекаємо завершення всього поточного батчу (разом із їхніми retry всередині)
        batch_results_lists = await asyncio.gather(*tasks)
        
        # Flatten (розгортаємо) списки відповідей в один загальний список
        for res_list in batch_results_lists:
            results.extend(res_list)
            
        # Якщо залишились ще батчі для обробки — чекаємо batch_delay
        if i + api_calls_per_batch < len(all_groups):
            logging.info(f"Batch completed. Waiting {batch_delay} seconds before sending the next batch...")
            await asyncio.sleep(batch_delay)
            
    return results


# ---------------------------------------------------------------------------
# 6. Entry Point
# ---------------------------------------------------------------------------

async def main():
    # --- Load environment variables from .env (GEMINI_API_KEY) ---
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY is not set. Copy .env.example to .env and fill in your key.")
        sys.exit(1)

    # --- Load CSV ---
    CSV_PATH = "input_requests.csv"
    try:
        df = load_requests(CSV_PATH)
    except Exception as exc:
        logging.error(f"Failed to load CSV: {exc}")
        sys.exit(1)
        
    if df.empty:
        logging.info("CSV is empty. Exiting.")
        return

    # --- Process all requests ---
    print(f"\n{'='*60}\nStarting processing of {len(df)} requests...\n{'='*60}\n")
    
    # Тут можна вибрати режим: "economy" (заощаджує запити) або "heavy" (по одному на виклик)
    results = await process_all_requests(
        df, 
        mode="economy",           # "economy" або "heavy"
        rows_per_call=5,          # Кількість рядків в одному запиті (тільки для economy)
        api_calls_per_batch=12,   # Скільки паралельних LLM-запитів ми робимо за раз
        batch_delay=75            # Затримка між батчами (або під час retry)
    )

    # --- Count successes ---
    successful_count = sum(1 for r in results if not r.get("_error"))
    
    print(f"\n{'='*60}")
    print(f"Processing completed.")
    print(f"Total requests: {len(df)}")
    print(f"Successfully parsed: {successful_count}")
    print(f"Failed (fallback used): {len(df) - successful_count}")
    print(f"{'='*60}\n")
    

if __name__ == "__main__":
    asyncio.run(main())
