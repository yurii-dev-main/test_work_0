# AI Request Classification Service

Сервіс для автоматичної класифікації бізнес-запитів з використанням Google Gemini LLM та Pydantic. 
Цей інструмент приймає сирі текстові запити (з CSV файлу), використовує штучний інтелект для їх аналізу, вилучення конкретних кроків, визначення пріоритету та цільового відділу, а потім валідує відповіді через суворі схеми Pydantic. Зрештою, сервіс формує звіти та інтегрується з Google Sheets та Telegram.

## 🚀 Як запустити

### Локальний запуск (Python)

1. Клонуйте репозиторій та перейдіть у директорію проекту.
2. Встановіть залежності:
   ```bash
   pip install -r requirements.txt
   ```
3. Скопіюйте `.env.example` у файл `.env` та заповніть свої облікові дані:
   ```bash
   cp .env.example .env
   ```
4. Покладіть файл з даними `input_requests.csv` у корінь проекту.
5. (Опціонально) Додайте ваш `google_credentials.json` для інтеграції з Google Sheets.
6. Запустіть скрипт:
   ```bash
   python main.py
   ```

### Запуск через Docker

1. Зберіть Docker-образ:
   ```bash
   docker build -t ai-classifier .
   ```
2. Запустіть контейнер. Зверніть увагу, що файли `input_requests.csv` та `google_credentials.json` мають бути прокинуті всередину контейнера через volume, так само як і файл `.env`.

   **Для Linux / macOS (Bash):**
   ```bash
   docker run --env-file .env -v $(pwd)/input_requests.csv:/app/input_requests.csv -v $(pwd)/google_credentials.json:/app/google_credentials.json ai-classifier
   ```

   **Для Windows (PowerShell):**
   ```powershell
   docker run --env-file .env -v "${PWD}/input_requests.csv:/app/input_requests.csv" -v "${PWD}/google_credentials.json:/app/google_credentials.json" ai-classifier
   ```

## ⚙️ Змінні оточення (.env)
- `GEMINI_API_KEY` — Ваш ключ API для Google Gemini.
- `TELEGRAM_BOT_TOKEN` — Токен вашого Telegram бота (для сповіщень).
- `TELEGRAM_CHAT_ID` — ID чату, куди відправляти звіт.
- `GOOGLE_CREDENTIALS_FILE` — Шлях до JSON-файлу сервісного акаунту Google.
- `SPREADSHEET_ID` — ID таблиці Google Sheets.

## ⚠️ Обмеження та Edge Cases

*   **Rate Limits & Batching:** Безкоштовний тариф Gemini API має жорсткий ліміт у 15 запитів на хвилину (RPM). Для вирішення цієї проблеми ми реалізували механізм батчингу (групування до 5 запитів в один LLM-виклик у режимі `economy`) та логіку розумних повторень (`Retry`) із затримкою (`sleep`) при отриманні помилки `429 RESOURCE_EXHAUSTED`. Це запобігає перериванню роботи скрипта.
*   **Validation (Pydantic):** Використання суворих схем Pydantic у поєднанні зі Structured Outputs від Gemini мінімізує "галюцинації" моделі. Якщо LLM повертає невірний формат, Pydantic викидає `ValidationError`, після чого скрипт ловить помилку через `try/except` і безпечно повертає фолбек-відповідь (`needs_clarification=True`, `category="поза скоупом"`), щоб додаток ніколи не падав.

