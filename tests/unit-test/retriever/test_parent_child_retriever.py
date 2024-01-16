import os
import pytest
from mock import patch, mock_open
from rag.retriever import ParentChildRetriever
import rag
from langchain.docstore.document import Document

class TestParentChildRetriever:

    @pytest.fixture(scope="class")
    @patch("builtins.open", mock_open())
    @patch("json.load")
    def pcr(self, mock_json_load):
        print("in config")
        pcr_test_db_path = "/tmp/pcr_test_db"
        if not os.path.exists(pcr_test_db_path):
            os.mkdir(pcr_test_db_path)
        mock_json_load.return_value = {"doc_store_path": pcr_test_db_path}
        rag.config.Config.init("")
        return ParentChildRetriever()
    
    def test_should_add_data(self, pcr):
        docs = [Document(page_content="parots are green in colour",
                        metadata={"source": "local"}),
                Document(page_content="Team 7 has Naruto, Sasuke and Sakura",
                        metadata={"source": "local"})]
        pcr.retriever.add_documents(docs)