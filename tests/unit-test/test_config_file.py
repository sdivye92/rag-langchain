import pytest
from mock import patch, mock_open
from rag.config import MilvusConfig, Config



class TestInitConfig:
    
    @patch("builtins.open", mock_open())
    @patch("json.load")
    def test_should_init_config(self, mock_json_load):
        mock_json_load.return_value = {}
        Config.init("")
        assert Config.get_instance() != None
    
    @patch("builtins.open", mock_open())
    @patch("json.load")
    def test_should_fail_milvus_collection_name(self, mock_json_load):
        collection_name = "Chunk Store"
        mock_json_load.return_value = {"milvus" : {"chunk_collection_name": collection_name}}
        with pytest.raises(ValueError,
                           match=MilvusConfig.error_msg.format(collection_name)):
            config = Config("")