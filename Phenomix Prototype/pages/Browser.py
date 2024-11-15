import streamlit as st
from neo4j import GraphDatabase

# Streamlit app configuration
st.set_page_config(page_title="Phenotype Browser", page_icon="🔍")

# Initialize Neo4j driver
@st.cache_resource
def init_driver():
    return GraphDatabase.driver(st.session_state.neo_uri, auth=(st.session_state.neo_user, st.session_state.neo_password))

driver = init_driver()

# Fetch phenotype data from the database
@st.cache_data
def fetch_phenotype_data():
    query = """
    MATCH (p:phenotype)
    RETURN p
    """
    with driver.session() as session:
        result = session.run(query)
        data = [record["p"] for record in result]
    return data

# Generate tags based on the record ID
def get_tags(record_id):
    tags = []
    if record_id[0].lower() == 's':
        tags.append('<span style="border-radius: 20px; padding: 4px 8px; margin-right: 4px; color: white; background-color: darkblue;">Sentinel</span>')
    if record_id[1].lower() == 'h':
        tags.append('<span style="border-radius: 20px; padding: 4px 8px; margin-right: 4px; color: white; background-color: teal;">HDRUK</span>')
    if record_id[2].lower() == 'c':
        tags.append('<span style="border-radius: 20px; padding: 4px 8px; margin-right: 4px; color: white; background-color: blue;">CPRD</span>')
    if record_id[3].lower() == 'o':
        tags.append('<span style="border-radius: 20px; padding: 4px 8px; margin-right: 4px; color: white; background-color: orange;">OHDSI</span>')
    if record_id[4].lower() == 'p':
        tags.append('<span style="border-radius: 20px; padding: 4px 8px; margin-right: 4px; color: white; background-color: green;">PheKB</span>')
    return ' '.join(tags)

def update_current_phenotype(phenotype_id):
    st.session_state['current_pheno'] = phenotype_id

def show():
    st.title('Phenotype Browser')

    # Fetch data
    data = fetch_phenotype_data()

    # Search bar
    search_query = st.text_input("Search for a phenotype:")
    filtered_data = [record for record in data if search_query.lower() in record['phenotypes'].lower()]

    # Display the number of phenotypes being shown
    st.markdown(f"Showing {len(filtered_data)} phenotypes out of {len(data)}")
    st.markdown("---")

    # Display filtered data
    for record in filtered_data:
        st.markdown(f"### {record['phenotypes']}")
        st.markdown(f"**ID:** {record['id']}")
        st.markdown(get_tags(record['id']), unsafe_allow_html=True)
        if st.button(f"Explore {record['phenotypes']} at *View Phenotype*", key=record['id']):
            update_current_phenotype(record['id'])
        st.markdown("---") 

show()







