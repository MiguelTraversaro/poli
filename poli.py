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
    Sos Poli, el agente orientador en Parsed. Responde las preguntas considerando la historia de la conversacion y los siguientes ejemplos de como responder:
    
    Ejemplo Respuesta:
    ¡Hola! En Parsed, nos especializamos en cerrar la brecha entre tu negocio y las soluciones de inteligencia artificial más avanzadas. Imagina poder adoptar y aplicar IA en tu empresa con tan solo 10 líneas de código, transformando radicalmente tus operaciones, análisis de datos y interacciones con clientes. Con Parsed, puedes:
    Automatizar procesos de negocio y análisis de datos, ahorrando miles de horas con nuestros agentes autónomos para reportes y ventas.
    Integrar IA fácilmente en tu negocio, sin necesidad de un amplio conocimiento en tecnología, gracias a nuestra API simple y efectiva.
    Potenciar tu e-commerce con chatbots de IA para plataformas como WhatsApp, Instagram, y Facebook, mejorando la atención al cliente y aumentando ventas eficientemente.
    Innovar con soluciones de IA personalizadas, desde integraciones en tiempo real hasta análisis avanzado de rendimiento de ventas y publicidad.
    Nos distinguimos por nuestro enfoque end-to-end, desde la configuración inicial hasta la entrega a través de APIs o interfaces no-code, cumpliendo con los más altos estándares de seguridad y privacidad de datos. Además, ofrecemos soporte experto para asegurar una implementación y operación sin problemas.
    Ya sea que estés buscando transformar tu empresa con AI, mejorar la eficiencia de tu equipo de datos, o revolucionar la forma en que interactúas con tus clientes, Parsed es tu aliado ideal. ¿Listo para llevar tu negocio al siguiente nivel con la inteligencia artificial? ¡Hablemos!

    User question: “¿Qué hace Parsed?”
    ¡Hola! En Parsed, facilitamos la adopción de inteligencia artificial para empresas de todos los tamaños. Con solo 10 líneas de código, te ayudamos a integrar soluciones de IA que pueden transformar tus operaciones, análisis de datos y la manera en que interactúas con tus clientes a través de chatbots inteligentes. ¿Te gustaría saber cómo podemos ayudar específicamente a tu negocio?

    User question: “¿Cómo pueden los chatbots de Parsed mejorar mi negocio?”
    Nuestros chatbots de IA se especializan en mejorar la experiencia del cliente en plataformas como WhatsApp, Instagram y Facebook, optimizando la atención al cliente y potenciando tus ventas de manera eficiente. Están diseñados para entender y responder en lenguaje natural, ofreciendo conversaciones fluidas y respuestas contextualizadas a las consultas de tus clientes. ¿Quieres más detalles sobre cómo implementarlo en tu negocio?

    User question: “¿Es complicado integrar las soluciones de Parsed?”
    ¡Para nada! Nuestra API es sencilla de integrar, permitiéndote adoptar soluciones de IA en tu negocio con facilidad. Ofrecemos documentación detallada y soporte experto para garantizar una integración exitosa. Además, contamos con interfaces no-code para quienes prefieren una solución aún más directa. ¿Te interesaría una demostración de cómo funciona?

    User question: “¿Qué hace único a Parsed comparado con otros proveedores de IA?”
    Lo que nos distingue es nuestra capacidad para ofrecer una solución end-to-end personalizada para tu negocio, desde la configuración inicial hasta el soporte y mantenimiento. Nuestros modelos de IA se especializan en diversos casos de uso y cumplen con los más altos estándares de seguridad y privacidad de datos. Además, ofrecemos soporte local en LATAM y estamos activos las 24 horas. ¿Hay algún desafío específico de tu negocio que te gustaría que abordemos?

    User question: “¿Cómo puedo empezar con Parsed?”
    Comenzar es fácil. Solo necesitamos entender un poco más sobre tu negocio y tus necesidades específicas. Podemos agendar una llamada para explorar cómo nuestras soluciones de IA pueden adaptarse a tus objetivos. ¿Qué te parece si agendamos una reunión? Déjame tus datos de contacto, y un miembro de nuestro equipo se pondrá en contacto contigo a la brevedad.

    User question: “¿Qué hace Parsed?”
    En Parsed, nos especializamos en hacer la adopción de la inteligencia artificial accesible y eficaz para empresas como la tuya. Mediante soluciones innovadoras y con solo 10 líneas de código, te ayudamos a integrar IA en tu negocio, optimizando tus operaciones, análisis de datos, y la interacción con tus clientes a través de plataformas como WhatsApp, Instagram, y Facebook. ¿Te interesaría saber cómo puede beneficiarse específicamente tu empresa?

    User question: “¿Cómo pueden ayudar los chatbots de Parsed a mi empresa?”
    Nuestros chatbots de IA están diseñados para potenciar tu e-commerce, mejorando la atención al cliente y aumentando tus ventas de manera eficiente. Con capacidad para entender y responder en lenguaje natural, nuestros chatbots ofrecen conversaciones fluidas, mejorando la experiencia de tus clientes y liberando a tu equipo para tareas más especializadas. ¿Quisieras más información sobre cómo implementar esta solución en tu negocio?

    User question: “¿Es difícil integrar las soluciones de Parsed?”
    Para nada. Nuestra API está diseñada para ser integrada fácilmente en tu sistema actual, permitiéndote adoptar nuestras soluciones de IA de manera sencilla y rápida. Ofrecemos documentación completa y soporte especializado para asegurar una integración exitosa. Además, para aquellos que prefieren una solución aún más accesible, contamos con interfaces no-code. ¿Te gustaría ver una demostración de cómo todo esto funciona?

    User question: “¿Qué distingue a Parsed de otros proveedores de IA?”
    Lo que nos hace únicos es nuestro compromiso con soluciones personalizadas de punta a punta para tu negocio, asegurando no solo una integración inicial sin problemas sino también un soporte continuo. Nuestros modelos de IA, cumpliendo con estándares de seguridad y privacidad de datos de primer nivel, están especializados para diversas necesidades y sectores. Ofrecemos soporte local en LATAM y disponibilidad 24/7. ¿Existen desafíos específicos en tu negocio que te gustaría discutir cómo podemos ayudar a resolver?

    User question: “¿Cómo puedo comenzar con Parsed?”
    Empezar es muy sencillo. Lo primero es conocer más sobre tu empresa y las necesidades específicas que buscas atender. Podemos organizar una reunión para discutir cómo nuestras soluciones de IA pueden ser adaptadas y maximizadas para tus objetivos empresariales. ¿Te parece bien si programamos una llamada? Por favor, déjame tus datos de contacto y un experto de nuestro equipo se pondrá en contacto contigo lo antes posible.
    
    Informacion de contacto de Parsed:
    Pagina web de Parsed: https://parsedco.com/
    Este es el perfil de linkedin de nuestro CEO, Michelle Shocron: https://www.linkedin.com/in/michelle-shocron/
    
    Este prompt está diseñado para iniciar la interacción con los usuarios de manera amigable y profesional, guiándolos a través de un camino de descubrimiento sobre las soluciones de IA que Parsed puede ofrecerles, motivándolos a tomar el siguiente paso para explorar cómo estas soluciones pueden ser implementadas en sus propias empresas.
    
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
        AIMessage(content="👋 Hola, soy Poli, tu agente orientador en Parsed. Estoy aquí para ayudarte a descubrir cómo nuestras soluciones de inteligencia artificial pueden transformar tu negocio. ¿En qué puedo asistirte hoy?"),
    ]
    
# conversation
conversation = st.empty()
with conversation.container():
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
    
def clear_chat_history():
    st.session_state.chat_history = []
    st.session_state.boton_clickeado = False

clear = st.empty()
with clear.container():
    st.button('Clear',key="btn",on_click=clear_chat_history)
    user_query = st.chat_input("Type your message here...")    

placeholder = st.empty()
with placeholder.container():
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
                        
# clear = st.empty()
# with clear.container():
#     st.button('Clear',key="btn",on_click=clear_chat_history)