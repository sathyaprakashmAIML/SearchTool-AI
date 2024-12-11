from langchain_community.tools import WikipediaQueryRun,DuckDuckGoSearchRun,ArxivQueryRun
from langchain.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_community.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent,AgentType
from langchain_groq import ChatGroq
import streamlit as st
load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

st.title("AI Chat with Integrated Search and Research Tools")
st.sidebar.title("API KEY")
api_key=st.sidebar.text_input("Enter you Groq API Key",type="password")
wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki_tool=WikipediaQueryRun(api_wrapper=wiki)

arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv_tool=ArxivQueryRun(api_wrapper=arxiv)

search=DuckDuckGoSearchRun(name="Search")

tools=[wiki_tool,arxiv_tool,search]

if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"Hi,I am a chat bot can search from web,How can i help you ?"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is Generative AI"):
    st.session_state.messages.append({'role':'user','content':prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(api_key=api_key,model="gemma2-9b-it",streaming=True)
    agent=initialize_agent(llm=llm,tools=tools,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    with st.spinner("Thinking..."):
            with st.chat_message('assistant'):
                 st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
                 response=agent.run(st.session_state.messages,callbacks=[st_cb])
                 st.session_state.messages.append({'role':"assistant",'content':response})
                 st.success(response)


        
    




