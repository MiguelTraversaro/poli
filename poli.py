import openai
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from string import Template

with st.sidebar:
    st.title('🤖💬 OpenAI Chatbot')
    if 'OPENAI_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
        openai.api_key = st.secrets['OPENAI_API_KEY']
    else:
        openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
        if not (openai.api_key.startswith('sk-') and len(openai.api_key)==51):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

poli_template = Template("""
                         
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

    Context: $context
            
    History: $history
            
    Human: $query
    Asistente de IA:""")

poli_prompt = """ You are POLI, the AI assistant of Parsed. """

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": m["role"], "content": m["content"]}
                      for m in st.session_state.messages], 
            stream=True,
            template=poli_template, 
            system_prompt=poli_prompt
            ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})