import warnings, traceback
warnings.simplefilter("error", UserWarning)
from app.llm import create_chat_llm
from app.agents.schemas import QueryAnalysis
llm = create_chat_llm()
try:
    for c in llm.with_structured_output(QueryAnalysis).stream("what is the leave policy?"):
        pass
    print("NO WARNING via .stream()")
except UserWarning:
    traceback.print_exc()
