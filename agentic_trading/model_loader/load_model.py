import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from agentic_trading.constant import *
from agentic_trading.custom_logging.my_logger import logger
from agentic_trading.utils.config import read_yaml
from agentic_trading.Exception.exception import TradingBotException
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import sys



class ModelLoader:
    def __init__(self,config_file_path=CONFIG_FILE_PATH):
        self.config=read_yaml(config_file_path)

        load_dotenv()
        self._validate_env()

    def _validate_env(self):
        """
        Ensure required environment variables are available.
        """
        required_vars = ["GOOGLE_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        
    def load_llm(self):
        try:
            config=self.config.Model_loader
            llm = ChatGoogleGenerativeAI(
                model=config.llm_model_name,
            )
            return llm
        except Exception as e:
            raise TradingBotException(e,sys)
    
    def load_embeddings(self):
        try:
            """
            Load Hugging Face embeddings for vector store.
            """
            logger.info("Loading Hugging Face embedding model...")
            config = self.config.Model_loader
            model_name = config.model_name
            huggingface_embeddings = HuggingFaceEmbeddings(model_name=model_name)
            logger.info(f"Embedding model {model_name} loaded successfully")
            return huggingface_embeddings
        except TradingBotException as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
# if __name__ == "__main__":
#     model=ModelLoader()
#     response=model.load_llm()
#     result=response.invoke("hi")
#     print(result.content)