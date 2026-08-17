import warnings, traceback
warnings.simplefilter("error", UserWarning)
from app.llm import create_chat_llm
from app.agents.schemas import QueryAnalysis
llm = create_chat_llm()
try:
    r = llm.with_structured_output(QueryAnalysis).invoke("what is the leave policy?")
    print("NO WARNING, result:", r)
except UserWarning:
    traceback.print_exc()
