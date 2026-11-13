import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "test-embedding-3-small")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-6-luna")
    REASONING_EFFORT: str = os.getenv("REASONING_EFFORT", "low")

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    )

    CHATBOT_COLLECTION: str = os.getenv("CHATBOT_COLLECTION", "company_policies")
    RECOMMEND_COLLECTION: str = os.getenv("RECOMMEND_COLLECTION", "product_catalog")


settings = Settings()
