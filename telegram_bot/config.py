from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class Settings(BaseSettings):
    TOKEN: str

    model_config = SettingsConfigDict(env_file='.env')


settings = Settings()