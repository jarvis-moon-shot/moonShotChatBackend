from langchain import HuggingFacePipeline, LLMChain
from transformers import AutoTokenizer
import transformers
import torch

def get_chain(
   question_handler, stream_handler, tracing: bool = False
) -> LLMChain:
    model = 'meta-llama/Llama-2-7b-chat-hf'
    # tokenizer=AutoTokenizer.from_pretrained(model)
    # pipeline=transformers.pipeline("text-generation",
    #                             model=model,
    #                             tokenizer=tokenizer,
    #                             torch_dtype=torch.bfloat15,
    #                             trust_remote_code=True,
    #                             device_map="auto",
    #                             max_length=1000,
    #                             do_sample=True,
    #                             top_k=10,
    #                             num_return_sequences=1,
    #                             eos_token_id=tokenizer.eos_token_id
    #                             )

    llm = HuggingFacePipeline.from_model_id(
    model_id=model,
    task="text-generation",
    # model_kwargs={"temperature": 0, "max_length": 64},
)

    # llm = HuggingFacePipeline(pipeline=pipeline)
    from langchain import PromptTemplate,LLMChain
    template = """ Write a summary of following text del;imited by triple quotes.
    '''{text}'''
    """

    prompt=PromptTemplate(template=template, input_variables=['text'])
    llm_chain=LLMChain(prompt=prompt, llm=llm)
    text = """About uttar pradesh"""
    print(llm_chain.run(text))
    return llm_chain