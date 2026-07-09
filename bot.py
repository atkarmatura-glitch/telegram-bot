"""
KXX AI Bot — Telegram uchun umumiy AI-chatbot

Ishlashi:
- Foydalanuvchi botga yozadi -> bot AI orqali javob qaytaradi
- Ikkita "provider" qo'llab-quvvatlanadi:
    * Gemini (Google) — BEPUL, sinov uchun tavsiya qilinadi
    * Claude (Anthropic) — pullik, sifatli, production uchun
- Qaysi birini ishlatishni AI_PROVIDER muhit o'zgaruvchisi orqali tanlaysiz
- Har bir foydalanuvchi uchun alohida suhbat tarixi saqlanadi (xotira)
- /start, /reset, /help buyruqlari mavjud
"""

import os
import logging
from collections import defaultdict

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ---------------------------------------------------------------------------
# Sozlamalar
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
AI_PROVIDER = os.environ.get("AI_PROVIDER", "gemini").lower()  # "gemini" yoki "claude"

SYSTEM_PROMPT = (
    "Sen KXX guruhi uchun do'stona, foydali AI-yordamchisan. "
    "Foydalanuvchilarga savollariga aniq va qisqa javob ber. "
    "Agar savol o'zbek tilida bo'lsa, o'zbek tilida javob ber."
)

MAX_HISTORY_MESSAGES = 20  # har bir foydalanuvchi uchun saqlanadigan xabarlar soni

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# user_id -> [{"role": "user"/"assistant", "content": "..."}]
conversations: dict[int, list[dict]] = defaultdict(list)


# ---------------------------------------------------------------------------
# AI provider — Gemini (bepul) yoki Claude (pullik)
# ---------------------------------------------------------------------------

if AI_PROVIDER == "gemini":
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    _model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)

    def ask_ai(history: list[dict]) -> str:
        # Gemini formatiga o'girish: "assistant" -> "model"
        gemini_history = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in history[:-1]
        ]
        chat = _model.start_chat(history=gemini_history)
        response = chat.send_message(history[-1]["content"])
        return (response.text or "").strip()

elif AI_PROVIDER == "claude":
    from anthropic import Anthropic

    _client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

    def ask_ai(history: list[dict]) -> str:
        response = _client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=history,
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

else:
    raise ValueError(f"Noma'lum AI_PROVIDER: {AI_PROVIDER!r} (gemini yoki claude bo'lishi kerak)")


# ---------------------------------------------------------------------------
# Buyruqlar
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    conversations.pop(update.effective_user.id, None)
    await update.message.reply_text(
        "Salom! Men AI-yordamchiman. Menga istalgan savolingizni yozing.\n\n"
        "Buyruqlar:\n"
        "/reset — suhbatni tozalash\n"
        "/help — yordam"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Shunchaki menga xabar yozing — men AI orqali javob beraman.\n"
        "/reset — suhbat xotirasini tozalaydi."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    conversations.pop(update.effective_user.id, None)
    await update.message.reply_text("Suhbat tarixi tozalandi. Yangidan boshlaymiz!")


# ---------------------------------------------------------------------------
# Asosiy xabar ishlovchisi
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_text = update.message.text

    history = conversations[user_id]
    history.append({"role": "user", "content": user_text})
    history = history[-MAX_HISTORY_MESSAGES:]

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        reply_text = ask_ai(history) or "Kechirasiz, javob shakllantirib bo'lmadi. Qayta urinib ko'ring."
    except Exception as exc:  # noqa: BLE001
        logger.exception("AI so'rovida xatolik")
        history.pop()  # xato bo'lsa, foydalanuvchi xabarini tarixdan olib tashlaymiz
        conversations[user_id] = history
        await update.message.reply_text(f"Kechirasiz, xatolik yuz berdi: {exc}")
        return

    history.append({"role": "assistant", "content": reply_text})
    conversations[user_id] = history[-MAX_HISTORY_MESSAGES:]

    await update.message.reply_text(reply_text)


# ---------------------------------------------------------------------------
# Ishga tushirish
# ---------------------------------------------------------------------------

def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot ishga tushdi... (AI_PROVIDER=%s)", AI_PROVIDER)
    app.run_polling()


if __name__ == "__main__":
    main()
