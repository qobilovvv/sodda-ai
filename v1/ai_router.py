from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from .catalog import Product
from .config import Settings


@dataclass(frozen=True)
class AiContext:
    user_text: str
    products: list[Product]
    brands: list[str] | None = None
    categories: list[str] | None = None
    has_more: bool = False
    history: list[tuple[str, str]] | None = None  # (role, text) where role in {"user","assistant","note"}
    last_shown_product_ids: list[int] | None = None
    mode: str = "chat"  # chat | clarify | results | pagination | list | out_of_scope


def _product_snippet(p: Product) -> str:
    price = "so'rov bo'yicha" if p.price_usd <= 0 else f"{int(p.price_usd)}$" if float(int(p.price_usd)) == float(p.price_usd) else f"{p.price_usd:.2f}$"
    bits = [f"id={p.id}", (p.title or "").strip() or f"Mahsulot #{p.id}", f"narx={price}"]
    if p.brand:
        bits.append(f"brend={p.brand}")
    if p.category:
        bits.append(f"kategoriya={p.category}")
    return " | ".join(bits)


def generate_ai_reply(settings: Settings, ctx: AiContext) -> str:
    client = OpenAI(api_key=settings.openai_api_key)

    products_block = "\n".join([f"- {_product_snippet(p)}" for p in ctx.products[:4]])
    brands_block = "\n".join([f"- {b}" for b in (ctx.brands or [])[:20]])
    categories_block = "\n".join([f"- {c}" for c in (ctx.categories or [])[:20]])

    system = (
        "Sen Sodda.uz savdo menejerisan.\n"
        "Qoidalar:\n"
        "- Doim quvnoq, iliq va insondek yoz.\n"
        "- 1-4 gapdan oshirma.\n"
        "- Har xabarda salomlashib ketma; faqat suhbat boshida yoki mijoz salomlashsa salomlash.\n"
        "- Agar mode=clarify bo'lsa: hech qachon 'topilmadi/mavjud emas' demagin. Faqat 2 ta qisqa savol ber (1) byudjet USD (2) brend/xohish yoki muhim 1-2 xususiyat.\n"
        "- Agar mode=list bo'lsa: brend/kategoriya ro'yxatini o'zing yozib chiqma. Faqat 'ro'yxatni yuboryapman' deb ayt.\n"
        "- Agar mode=out_of_scope bo'lsa: mahsulot yo'q deb ayt va hech qanday aniqlashtiruvchi savol bermasdan, bizdagi yo'nalishlar (maishiy texnika va h.k.)ga yo'naltir.\n"
        "- Agar mijoz aniq aytmagan bo'lsa, 1 ta aniq savol ber (byudjet/kategoriya/parametr).\n"
        "- Mijoz hali mahsulot aytmagan bo'lsa, aniq mahsulot turlarini taxmin qilib misol keltirma; faqat umumiy yo'naltir: “mahsulot nomi yoki kategoriya yozing”.\n"
        "- Narx/ma'lumotni uydirma qilma: faqat berilgan ro'yxatga tayan.\n"
        "- Agar 'Topilgan mahsulotlar' bo'limida mahsulotlar bo'lsa, 'topa olmadim/mavjud emas' deb yozma.\n"
        "- Mijoz mahsulot so'rasa: 'mana mos variantlar' deb ayt, keyin bot mahsulot kartalarini yuboradi.\n"
        "- Agar hozir 'Topilgan mahsulotlar: yo'q' bo'lsa, 'mavjud emas' deb aytma; shunchaki topa olmaganingni ayt va 2 ta qisqa aniqlashtiruvchi savol ber.\n"
        "- Agar brendlar yoki kategoriyalar ro'yxati berilgan bo'lsa, mode=list bo'lmaganda javobda ularni ro'yxat qilib yoz (qisqa bulletlar bilan).\n"
        "- Mijoz aytmagan brend/byudjet/xususiyatlarni tilga olma (masalan, 'Samsung 500$' kabi taxmin qilma).\n"
        "- Telefon raqamini umuman yozma (hech qachon). Telefon faqat bot yuboradigan mahsulot kartalarida bo'ladi.\n"
        "- Agar mode=pagination bo'lsa: 'Yana variantlar bor' deb ayt va mijozdan 'yana' deb yozishini so'ra. Hech qachon 'topilmadi' demagin.\n"
        "Til: o'zbek (lotin)."
    )

    history_lines: list[str] = []
    for role, text in (ctx.history or [])[-10:]:
        if not text:
            continue
        prefix = "Mijoz" if role == "user" else "Menejer" if role == "assistant" else "Eslatma"
        history_lines.append(f"{prefix}: {text}")
    history_block = "\n".join(history_lines)

    last_shown = ", ".join([str(i) for i in (ctx.last_shown_product_ids or [])[:8]])

    user = (
        (f"Suhbat konteksti (oxirgi xabarlar):\n{history_block}\n\n" if history_block else "")
        + (f"Oxirgi ko'rsatilgan mahsulot IDlari: {last_shown}\n\n" if last_shown else "")
        + f"Mode: {ctx.mode}\n\n"
        + f"Mijoz yozdi: {ctx.user_text}\n\n"
        + (f"Topilgan mahsulotlar (kartalar keyin yuboriladi):\n{products_block}\n\n" if products_block else "Topilgan mahsulotlar: yo'q\n\n")
        + (f"Brendlar (agar kerak bo'lsa ro'yxat):\n{brands_block}\n\n" if brands_block else "")
        + (f"Kategoriyalar (agar kerak bo'lsa ro'yxat):\n{categories_block}\n\n" if categories_block else "")
        + ("Yana mahsulotlar bor: HA\n" if ctx.has_more else "Yana mahsulotlar bor: YO'Q\n")
    )

    resp = client.responses.create(
        model=settings.openai_model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_output_tokens=180,
    )
    return (resp.output_text or "").strip() or "Marhamat, yordam beraman."
