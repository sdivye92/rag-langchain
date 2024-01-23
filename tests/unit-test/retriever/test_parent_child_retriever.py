import os
from unittest.mock import Mock, MagicMock

import pytest
from langchain.retrievers import ParentDocumentRetriever
from mock import patch, mock_open

from rag.db_connection_handler._base_db_connector_builder import OffMemoryDatabaseConnectorBuilder
from rag.retriever import ParentChildRetriever
import rag
from langchain.docstore.document import Document
from rag.config import Config


class TestParentChildRetriever:

    @pytest.fixture(scope="class")
    @patch("builtins.open", mock_open())
    @patch("json.load")
    @patch.object(ParentDocumentRetriever, "__init__")
    def pcr(self, mock_parent_document_retriever, mock_json_load):
        mock_parent_document_retriever.return_value = None
        config_instance = MagicMock()
        config_instance.doc_store_path = "/some_path"
        config_instance.milvus.chunk_collection_name = "my_collection"

        mock_db_connector = MagicMock()
        mock_db_connector.get_connector = Mock(return_value="")

        pcr_test_db_path = "/tmp/pcr_test_db"
        if not os.path.exists(pcr_test_db_path):
            os.mkdir(pcr_test_db_path)
        mock_json_load.return_value = {"doc_store_path": pcr_test_db_path}
        rag.config.Config.init("")
        return ParentChildRetriever(config=config_instance, db_connector_builder=mock_db_connector)

    @patch.object(ParentDocumentRetriever, "add_documents")
    def test_should_add_data(self, mock_add_documents, pcr):
        docs = [Document(page_content="parots are green in colour",
                         metadata={"source": "local"}),
                Document(page_content="Team 7 has Naruto, Sasuke and Sakura",
                         metadata={"source": "local"})]
        pcr.add_documents(docs)
        mock_add_documents.assert_called_once_with(docs, None, True)
