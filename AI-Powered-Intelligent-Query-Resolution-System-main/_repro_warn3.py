import warnings, traceback
warnings.simplefilter("error", UserWarning)
from app.llm import create_chat_llm
from app.agents.schemas import QueryAnalysis
llm = create_chat_llm().model_copy(update={"disable_streaming": True})
try:
    out = None
    for c in llm.with_structured_output(QueryAnalysis).stream("what is the leave policy?"):
        out = c
    print("OK, no warning. parsed ->", type(out).__name__, out.is_clear)
except UserWarning:
    traceback.print_exc()
