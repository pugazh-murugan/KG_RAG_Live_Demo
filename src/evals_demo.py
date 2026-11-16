import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

load_dotenv()

# 1. Define a simple LangChain RAG generation chain
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
prompt = ChatPromptTemplate.from_template(
    "Answer the question using ONLY this context:\nContext: {context}\nQuestion: {question}"
)
rag_chain = prompt | llm | StrOutputParser()

# 2. Sample input & retrieved graph/document context
user_question = "Who founded TechNova Corp?"
retrieved_context = [
    "TechNova Corp is an AI company founded in 2018 by Dr. Elena Vance and Marcus Sterling."
]

# Run the LangChain chain to get the actual output
actual_output = rag_chain.invoke({
    "context": "\n".join(retrieved_context),
    "question": user_question
})
print(f"LangChain Output: {actual_output}\n")

# 3. Create a DeepEval Test Case
test_case = LLMTestCase(
    input=user_question,
    actual_output=actual_output,
    retrieval_context=retrieved_context
)

# 4. Define Evaluation Metrics (Threshold 0.7 = 70% pass score)
faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-5.4-mini")
relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-5.4-mini")

# 5. Run the evaluation
print("Running DeepEval metrics...")
evaluate(test_cases=[test_case], metrics=[faithfulness, relevancy])