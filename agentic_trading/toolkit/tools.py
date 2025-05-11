import os
import sys
from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_community.tools.polygon.financials import PolygonFinancials
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.tools.bing_search import BingSearchResults
from agentic_trading.Exception.exception import TradingBotException
from agentic_trading.model_loader.load_model import ModelLoader
from langchain_pinecone import PineconeVectorStore
from agentic_trading.utils.config import read_yaml
from agentic_trading.data_model.models import RagToolSchema
from agentic_trading.constant import *
from dotenv import load_dotenv
from pinecone import Pinecone


class ToolManager:
    def __init__(self,config_file_path=CONFIG_FILE_PATH):
        self.config=read_yaml(config_file_path)
        self.model_loader=ModelLoader()
        self.api_wrapper=PolygonAPIWrapper()
        load_dotenv()


    @tool(args_schema=RagToolSchema)
    def retriever_tool(self,question):
        """this is retriever tool"""
        config=self.config.tools
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=pinecone_api_key)
        vector_store = PineconeVectorStore(index=pc.Index(config.index_name), 
                                embedding= self.model_loader.load_embeddings())
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": config.top_k , "score_threshold": config.score_threshold},
        )
        retriever_result=retriever.invoke(question)
        
        return retriever_result
    
    def tavily(self):
        """tavily search from web"""
        config=config=self.config.tools
        tavilytool = TavilySearchResults(
            max_results=config.max_results,
            search_depth=config.search_depth,
            include_answer=True,
            include_raw_content=True,
            )
        return tavilytool
    
    def polygonfin(self):
        "it gives real time stock informations"
        financials_tool = PolygonFinancials(api_wrapper=self.api_wrapper)
        return financials_tool
    
    def tools(self):
        tools=[
            self.retriever_tool,
            self.polygonfin,
            self.tavily
        ]
        return tools