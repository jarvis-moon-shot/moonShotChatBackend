"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI



def get_chain(
   question_handler, stream_handler, tracing: bool = False
) -> LLMChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
        openai_api_key="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"
    )
    streaming_llm = OpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
       openai_api_key="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"
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager, output_key="context"
    )
    doc_chain = LLMChain(
        llm=streaming_llm, prompt=QA_PROMPT, callback_manager=manager
    )

    from langchain.chains import SequentialChain
    overall_chain = SequentialChain(chains=[question_generator, doc_chain], input_variables=['chat_history', 'question'], verbose=True)
    return overall_chain
