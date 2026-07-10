"""
KXX AI Bot — Telegram uchun umumiy AI-chatbot

Ishlashi:
- Foydalanuvchi botga yozadi -> bot AI orqali javob qaytaradi
- Rasm yuborilsa -> bot uni tahlil qiladi (vision)
- Ikkita "provider" qo'llab-quvvatlanadi:
    * Gemini (Google) — BEPUL, sinov uchun tavsiya qilinadi
    * Claude (Anthropic) — pullik, sifatli, production uchun
- Har bir foydalanuvchi uchun alohida suhbat xotirasi saqlanadi
- Admin uchun: /stats (statistika), /broadcast (xabar tarqatish)
- /start, /reset, /help buyruqlari mavjud
"""

import os
import io
import base64
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

# Admin ID(lar) - vergul bilan ajratilgan Telegram user ID(lar).
# Masalan: ADMIN_IDS=123456789,987654321
_admin_ids_raw = os.environ.get("ADMIN_IDS", "").strip()
ADMIN_IDS = {int(x) for x in _admin_ids_raw.split(",") if x.strip().isdigit()}

SYSTEM_PROMPT = (
    "Sen KXX guruhi uchun do'stona, foydali AI-yordamchisan. "
    "Foydalanuvchilarga savollariga aniq va qisqa javob ber. "
    "Agar savol o'zbek tilida bo'lsa, o'zbek tilida javob ber; "
    "boshqa tilda yozilsa, o'sha tilda javob ber."
)

VISION_DEFAULT_PROMPT = (
    "Ushbu rasmda nima ko'rsatilganini batafsil tasvirlab ber. "
    "Agar bu grafik/chart bo'lsa, undagi asosiy tendensiyalarni tushuntir. "
    "O'zbek tilida javob ber."
)

MAX_HISTORY_MESSAGES = 20

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# user_id -> [{"role": "user"/"assistant", "content": "..."}]
conversations: dict[int, list[dict]] = defaultdict(list)

# Botdan foydalangan barcha foydalanuvchilar (statistika/broadcast uchun).
# Eslatma: bu xotirada saqlanadi, server qayta ishga tushganda tozalanadi.
known_users: dict[int, str] = {}  # user_id -> username yoki ism


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


def remember_user(update: Update) -> None:
    user = update.effective_user
    if user:
        known_users[user.id] = user.username or user.first_name or str(user.id)


# ---------------------------------------------------------------------------
# AI provider — matn va rasm (vision) uchun
# ---------------------------------------------------------------------------

if AI_PROVIDER == "gemini":
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    _model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)

    def ask_ai(history: list[dict]) -> str:
        gemini_history = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in history[:-1]
        ]
        chat = _model.start_chat(history=gemini_history)
        response = chat.send_message(history[-1]["content"])
        return (response.text or "").strip()

    def ask_ai_vision(prompt: str, image_bytes: bytes, mime_type: str) -> str:
        import PIL.Image
        image = PIL.Image.open(io.BytesIO(image_bytes))
        response = _model.generate_content([prompt, image])
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

    def ask_ai_vision(prompt: str, image_bytes: bytes, mime_type: str) -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        response = _client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
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
    remember_user(update)
    conversations.pop(update.effective_user.id, None)
    await update.message.reply_text(
        "Salom! Men AI-yordamchiman. Menga istalgan savolingizni yozing "
        "yoki rasm yuboring — tahlil qilib beraman.\n\n"
        "Buyruqlar:\n"
        "/reset — suhbatni tozalash\n"
        "/help — yordam"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    remember_user(update)
    text = (
        "Shunchaki menga xabar yozing — men AI orqali javob beraman.\n"
        "Rasm yuborsangiz, uni tahlil qilib beraman.\n"
        "/reset — suhbat xotirasini tozalaydi."
    )
    if is_admin(update.effective_user.id):
        text += "\n\nAdmin buyruqlari:\n/stats — statistika\n/broadcast <xabar> — hammaga xabar"
    await update.message.reply_text(text)


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    remember_user(update)
    conversations.pop(update.effective_user.id, None)
    await update.message.reply_text("Suhbat tarixi tozalandi. Yangidan boshlaymiz!")


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Bu buyruq faqat admin uchun.")
        return
    await update.message.reply_text(
        f"📊 Statistika\n"
        f"Jami foydalanuvchilar (server ishga tushgandan beri): {len(known_users)}\n"
        f"Faol suhbatlar: {len(conversations)}"
    )


async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Bu buyruq faqat admin uchun.")
        return
    text = " ".join(context.args) if context.args else ""
    if not text:
        await update.message.reply_text("Foydalanish: /broadcast Sizning xabaringiz matni")
        return

    sent, failed = 0, 0
    for uid in list(known_users.keys()):
        try:
            await context.bot.send_message(chat_id=uid, text=text)
            sent += 1
        except Exception:
            failed += 1
    await update.message.reply_text(f"Yuborildi: {sent}, xato: {failed}")


# ---------------------------------------------------------------------------
# Xabar ishlovchilari
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    remember_user(update)
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
        history.pop()
        conversations[user_id] = history
        await update.message.reply_text(f"Kechirasiz, xatolik yuz berdi: {exc}")
        return

    history.append({"role": "assistant", "content": reply_text})
    conversations[user_id] = history[-MAX_HISTORY_MESSAGES:]

    await update.message.reply_text(reply_text)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    remember_user(update)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    photo = update.message.photo[-1]
    tg_file = await photo.get_file()
    image_bytes = bytes(await tg_file.download_as_bytearray())
    prompt = update.message.caption or VISION_DEFAULT_PROMPT

    try:
        reply_text = ask_ai_vision(prompt, image_bytes, "image/jpeg") or "Rasmni tahlil qilib bo'lmadi."
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vision so'rovida xatolik")
        await update.message.reply_text(f"Kechirasiz, rasmni tahlil qilishda xatolik: {exc}")
        return

    await update.message.reply_text(reply_text)


# ---------------------------------------------------------------------------
# Ishga tushirish
# ---------------------------------------------------------------------------

def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("broadcast", broadcast))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot ishga tushdi... (AI_PROVIDER=%s, admins=%s)", AI_PROVIDER, ADMIN_IDS)
    app.run_polling()


if __name__ == "__main__":
    main()
