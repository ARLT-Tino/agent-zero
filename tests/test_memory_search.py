import sys
import types

# Stub out heavy dependencies before importing memory
for mod in [
    "langchain.storage",
    "langchain.embeddings",
    "langchain_community.vectorstores",
    "langchain_community.docstore.in_memory",
    "langchain_community.vectorstores.utils",
    "langchain_community.document_loaders",
    "langchain_core.embeddings",
    "langchain_core.documents",
    "langchain_core.language_models.chat_models",
    "langchain_core.outputs.chat_generation",
    "langchain_core.callbacks.manager",
    "langchain_core.messages",
    "langchain_core",
    "faiss",
    "numpy",
    "python.helpers.faiss_monkey_patch",
    "webcolors",
    "html",
    "agent",
    "nest_asyncio",
    "litellm",
    "dotenv",
    "tiktoken",
    "sentence_transformers",
]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

sys.modules["langchain.storage"].InMemoryByteStore = object
sys.modules["langchain.storage"].LocalFileStore = object
sys.modules["langchain.embeddings"].CacheBackedEmbeddings = object
base_mod = types.ModuleType("langchain.embeddings.base")
base_mod.Embeddings = object
sys.modules["langchain.embeddings.base"] = base_mod

class _StubFaiss:
    pass

sys.modules["langchain_community.vectorstores"].FAISS = _StubFaiss
sys.modules["langchain_community.docstore.in_memory"].InMemoryDocstore = object
sys.modules["langchain_community.vectorstores.utils"].DistanceStrategy = object
sys.modules["langchain_community.document_loaders"].CSVLoader = object
sys.modules["langchain_community.document_loaders"].JSONLoader = object
sys.modules["langchain_community.document_loaders"].PyPDFLoader = object
sys.modules["langchain_community.document_loaders"].TextLoader = object
sys.modules["langchain_community.document_loaders"].UnstructuredHTMLLoader = object
sys.modules["langchain_community.document_loaders"].UnstructuredMarkdownLoader = object
sys.modules["langchain_core.embeddings"].Embeddings = object
sys.modules["langchain_core.documents"].Document = object

sys.modules["faiss"].IndexFlatIP = object
sys.modules["python.helpers.faiss_monkey_patch"].__dict__.update({})
sys.modules["numpy"].array = lambda x: x
# langchain_core stubs
sys.modules["langchain_core"].__dict__.update({})
sys.modules["langchain_core.language_models.chat_models"].SimpleChatModel = object
sys.modules["langchain_core.outputs.chat_generation"].ChatGenerationChunk = object
sys.modules["langchain_core.callbacks.manager"].CallbackManagerForLLMRun = object
sys.modules["langchain_core.callbacks.manager"].AsyncCallbackManagerForLLMRun = object
sys.modules["langchain_core.messages"].BaseMessage = object
sys.modules["langchain_core.messages"].AIMessageChunk = object
sys.modules["langchain_core.messages"].HumanMessage = object
sys.modules["langchain_core.messages"].SystemMessage = object
# Minimal Agent stub to satisfy type hints
agent_stub = types.ModuleType("agent")
class _Agent:
    def __init__(self):
        pass
agent_stub.Agent = _Agent
sys.modules["agent"] = agent_stub
sys.modules["nest_asyncio"].apply = lambda: None
sys.modules["litellm"].completion = lambda *a, **k: None
sys.modules["litellm"].acompletion = lambda *a, **k: None
sys.modules["litellm"].embedding = lambda *a, **k: None
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
sys.modules["tiktoken"].get_encoding = lambda *a, **k: None
sys.modules["sentence_transformers"].SentenceTransformer = object

from python.helpers.memory import Memory  # noqa: E402
from python.helpers.log import Log  # noqa: E402

class DummyDB:
    async def asearch(self, *args, **kwargs):
        raise ValueError("corrupted id")

class DummyAgent:
    def __init__(self):
        self.config = types.SimpleNamespace(memory_subdir="test", knowledge_subdirs=[], embeddings_model=None)
        self.context = types.SimpleNamespace(log=Log())
    async def rate_limiter(self, *args, **kwargs):
        pass

def test_search_similarity_handles_value_error(monkeypatch):
    async def run_test():
        agent = DummyAgent()
        mem = Memory(agent=agent, db=DummyDB(), memory_subdir="test")

        reload_called = False

        async def fake_reload(a):
            nonlocal reload_called
            reload_called = True
            return mem

        monkeypatch.setattr(Memory, "reload", fake_reload)

        results = await mem.search_similarity_threshold("q", limit=5, threshold=0.5)
        assert results == []
        assert reload_called
        assert any(item.type == "error" for item in agent.context.log.logs)

    import asyncio

    asyncio.run(run_test())
