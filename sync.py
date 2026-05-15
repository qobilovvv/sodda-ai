import pymysql.cursors
import pymysql
import os
import json
import shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()


# Helper function to run any query cleanly
def fetch_data_from_mysql(query):
    connection = pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_DATABASE"),
        cursorclass=pymysql.cursors.DictCursor
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    finally:
        connection.close()


def clean_json_field(data, lang_key="uz"):
    if not data:
        return ""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data

    if isinstance(data, dict):
        # Prefer Uzbek, fallback to Russian, then anything else
        content = data.get('uz', data.get('ru', next(iter(data.values())) if data else ""))
        if isinstance(content, dict):
            return ", ".join([f"{k}: {v}" for k, v in content.items()])
        return str(content)
    return str(data)


def extract_image_urls(image_data):
    if not image_data:
        return "None"

    try:
        # If MySQL returns it as a string, parse it into a Python list
        if isinstance(image_data, str):
            imgs = json.loads(image_data)
        else:
            imgs = image_data  # If pymysql already parsed the JSON

        # Check if it's a list and has items
        if isinstance(imgs, list) and len(imgs) > 0:
            urls = []
            for img in imgs:
                # Get the base filename (UUID), stripping any accidental paths
                clean_name = img.replace("products/", "").replace("thumbs/", "").strip("/")

                # We format it as a standard URL
                urls.append(f"https://sodda.uz/storage/products/{clean_name}")

            return ",".join(urls)

    except Exception as e:
        print(f"Image parsing error: {e}")

    return "None"


def sync_vector_db():
    print("🔄 Fetching data from database...")
    documents = []

    # 1. FETCH & PROCESS PRODUCTS
    print("📦 Processing Products...")
    products_query = """
        SELECT 
            p.id, p.title, p.sm_desc, p.spec, p.price, p.keywords, p.model, p.stock, 
            p.images, p.images_thumb,
            c.title AS category_title,   
            b.title AS brand_title      
        FROM products p
        LEFT JOIN categories c ON p.category_id = c.id
        LEFT JOIN brands b ON p.brand_id = b.id
        WHERE p.deleted_at IS NULL AND p.status = 1
    """
    products = fetch_data_from_mysql(products_query)

    for p in products:
        title = clean_json_field(p['title'])
        brand = clean_json_field(p['brand_title'])
        category = clean_json_field(p['category_title'])
        specs = clean_json_field(p['spec'])
        description = clean_json_field(p['sm_desc'])
        image_links = extract_image_urls(p['images'])

        # Notice the added TYPE: PRODUCT
        page_content = (
            f"TYPE: PRODUCT\n"
            f"PRODUCT_TITLE: {title}\n"
            f"MODEL: {p['model']}\n"
            f"BRAND: {brand}\n"
            f"CATEGORY: {category}\n"
            f"PRICE: {p['price']} UZS\n"
            f"IMAGE_LINKS: {image_links}\n"
            f"CHARACTERISTICS: {specs}\n"
            f"DESCRIPTION: {description}\n"
            f"KEYWORDS: {p['keywords']}"
        )

        metadata = {
            "entity_type": "product",
            "product_id": p['id'],
            "price": float(p['price']) if p['price'] else 0.0,
            "stock": p['stock']
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    # 2. FETCH & PROCESS BRANDS
    print("🏢 Processing Brands...")
    brands_query = "SELECT id, title, about FROM brands WHERE status = 1"
    brands = fetch_data_from_mysql(brands_query)

    for b in brands:
        title = b['title'] if b['title'] else ""
        about = clean_json_field(b['about'])

        page_content = (
            f"TYPE: BRAND\n"
            f"BRAND_NAME: {title}\n"
            f"ABOUT: {about}\n"
            f"INFO: Ushbu brend do'konimizda mavjud va uning mahsulotlarini sotamiz."
        )
        metadata = {
            "entity_type": "brand",
            "brand_id": b['id']
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    # 3. FETCH & PROCESS CATEGORIES
    print("📂 Processing Categories...")
    categories_query = "SELECT id, title FROM categories WHERE status = 1"
    categories = fetch_data_from_mysql(categories_query)

    for c in categories:
        title = clean_json_field(c['title'])

        page_content = (
            f"TYPE: CATEGORY\n"
            f"CATEGORY_NAME: {title}\n"
            f"INFO: Do'konimizda ushbu kategoriyadagi mahsulotlar sotiladi."
        )
        metadata = {
            "entity_type": "category",
            "category_id": c['id']
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    # Clear existing vector DB to prevent duplicates
    if os.path.exists("./chroma_db"):
        print("🗑 Clearing existing vector database...")
        shutil.rmtree("./chroma_db")

    print(f"🧠 Indexing {len(documents)} total documents to Chroma...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ Sync Complete!")


if __name__ == "__main__":
    sync_vector_db()