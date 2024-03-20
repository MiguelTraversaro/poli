import openai
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# app config
st.set_page_config(page_title="POLI 🤖💬",page_icon="🤖")
st.header('🤖💬 POLI - Parsed Chatbot',divider="rainbow")

# main function
def get_response(user_question, chat_history):
    template = """
    You are Poli, a helpful assistant at Parsed. Answer the following questions considering the history of the conversation and the following text:
    
    Parsed is a company built in order to bring efficiency in business operations of companies. We provide the alignment between business stakeholders, AI and employees, working as an AI core to produce autonomous agents. 
    Our objective is to enable companies to create 10x use cases from the same dataset, being able to have fully automated systems which enable for automations to take place.
    From the beginning we helped companies customize AI open source models to leverage from using the amount of information they have available in daily uses, at the end we helped them choose between having internal bots, external bots or automations. 
    Our mission is to onboard companies into AI by providing initial workshops that change the way employees think to leverage on AI. 

    We provide the no code workspace to create intuitive pre-trained AI models with company’s data and build 10x use cases with the same set.

    We help business include AI in their daily workload by providing a platform to centralize creations, prompts, interface to chat and integrate your context while enabling collaboration. This creates synergies to increase the productivity of your employees because they have the place to learn how to make AI useful in manual tasks by building autonomus agents that perform independent operations in a safe environment.

    Building the rails to enable prompt intelligence for business operations of the future. 

    This is our CEO linkedin profile 
    https://www.linkedin.com/in/michelle-shocron/ 
    https://parsedco.com/ 

    Introducing Parsed: Empowering Organizations with Collaborative AI and Knowledge Sharing

    Unlock the potential of artificial intelligence and revolutionize your organization's operations with Parsed—a leading SaaS subscription business built on a powerful web app platform. Our mission is to help teams harness the capabilities of generative AI while facilitating seamless knowledge sharing within your organization.

    At Parsed, we understand the transformative power of new technologies and the immense opportunities they present. Our platform provides teams with a user-friendly interface that shares the characteristics of traditional Business Intelligence (BI) interfaces. However, unlike traditional BI tools, our platform is designed specifically for non-technical profiles, enabling them to leverage AI as an autonomous agent.

    Our unique offering lies in empowering organizations to customize and harness the power of proprietary data through self-service AI agents. By leveraging AI, these agents automate a new universe of operations, providing companies with a competitive edge in the future of work. Parsed not only restores intellectual property ownership to the company but also fosters team collaboration by enabling traceability of colleagues' actions and encouraging the sharing of valuable use cases.

    With Parsed, you can expect tangible benefits for your organization. Our platform provides vetted ratings of average AI performance, giving you insights into the capabilities and limitations of AI models. This information is invaluable for making informed decisions, optimizing workflows, and maximizing workforce productivity. Studies have shown that organizations using Parsed experience an average productivity increase of at least 14%.

    Join us on the forefront of AI innovation, and unleash the true potential of your organization. With Parsed, collaborative AI and knowledge sharing are within reach like never before. Experience a new era of automation, efficiency, and growth.

    We enable teams to co-work with artificial intelligence, developing unique, hyper-customised solutions that enable productivity gains in up to 66% of work areas. We specialise in developing customised use cases guiding the client to solve high priority pain points by leveraging AI. We create solutions that emulate the process that would run in real time to show our clients the performance of AI to solve the pain point that the client builds confidence with the implementation of emerging technologies in high priority processes within the organisation.
    
    User question: {user_question}
    Chat history: {chat_history}
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(openai_api_key=openai.api_key,model="gpt-3.5-turbo-0125",temperature=0.8,model_kwargs={"top_p":0.8})
        
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "chat_history": chat_history,
        "user_question": user_question,
    })
    
if 'boton_clickeado' not in st.session_state:
    st.session_state.boton_clickeado = False

opciones_mensajes = [
    "¿Cómo revoluciona Parsed las operaciones empresariales?",
    "¿Qué capacitación ofrece Parsed para el uso de la IA?",
    "¿Cómo cambia Parsed la visión sobre IA?",
    "¿Parsed hace fácil usar IA sin ser experto?",
]    

# session state (chat history)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="👋 Hola! Soy Poli, el asistente de Parsed. ¿En qué puedo ayudarte?"),
    ]

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("📎"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("👨‍💻"):
            st.write(message.content)

def click(string):
    st.session_state.chat_history.append(HumanMessage(content=string))
    response = get_response(string, st.session_state.chat_history)
    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.boton_clickeado = True
    botones_placeholder.empty()
    
user_query = st.chat_input("Type your message here...")            
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("👨‍💻"):
        st.markdown(user_query)
    with st.chat_message("📎"):
        response = get_response(user_query, st.session_state.chat_history)
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))
elif not st.session_state.boton_clickeado:
    botones_placeholder = st.empty()
    with botones_placeholder.container():
        cols = st.columns(2)
        for i, opcion in enumerate(opciones_mensajes):
            with cols[i % 2]:
                if st.button(opcion, key=f"button_{i}",use_container_width=True,on_click=click,args=[opcion]):
                    # Limpiar el marcador de posición para hacer desaparecer los botones
                    botones_placeholder.empty()