import streamlit as st


def init_session_state():

    st.session_state.neo_uri = "neo4j+s://4bd45e1f.databases.neo4j.io"
    st.session_state.neo_user = "neo4j"
    st.session_state.neo_password = "UnK9EUt8kRSUwIOfLrOmj3d8ZjcB5dtsWA2OcOvgwj8"

    st.session_state.current_pheno = "XHCOP0159"


st.set_page_config(
    page_title="Home",
    page_icon="👋",
)



def show():
    # Main content for the app
    st.title(':blue[Pheno]mix')
    st.markdown(""" 
        Explore the Sentinel, HDRUK, CPRD, OHDSI, and PHEKB phenotype databases with browser as well as GPT + Neo4j integrated chatbot.
    """)


init_session_state()
show()

