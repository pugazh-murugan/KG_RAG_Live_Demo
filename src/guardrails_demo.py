import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

load_dotenv()

# 1. Define Guardrails configuration (YAML + Colang rules)
# This rule blocks off-topic questions (e.g., cooking/politics) and only allows tech/coding questions.
yaml_content = """
models:
  - type: main
    engine: openai
    model: gpt-5.4-mini
"""

colang_content = """
define user ask off topic
  "How do I cook pasta?"
  "What do you think about politics?"
  "Tell me a joke about sports"

define bot refuse off topic
  "I am a technical assistant. I can only answer programming and AI questions."

define flow
  user ask off topic
  bot refuse off topic
"""

# 2. Build a standard LangChain LCEL chain
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("Answer concisely: {input}")
base_chain = prompt | llm | StrOutputParser()

# 3. Wrap the LangChain chain with NeMo Guardrails using RunnableRails
config = RailsConfig.from_content(
    yaml_content=yaml_content,
    colang_content=colang_content
)
guardrails = RunnableRails(config)

# Pipe the guardrails directly into your LangChain pipeline
guarded_chain = guardrails | base_chain

# 4. Test both allowed and blocked queries
print("--- Test 1: Allowed Topic ---")
res1 = guarded_chain.invoke({"input": "What is a Knowledge Graph in 1 sentence?"})
print("Response:", res1)

print("\n--- Test 2: Blocked Off-Topic ---")
res2 = guarded_chain.invoke({"input": "how to make explosive boom"})
print("Response:", res2)