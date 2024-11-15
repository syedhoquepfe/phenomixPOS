import streamlit as st
from neo4j import GraphDatabase
from ast import literal_eval
import re
import pandas as pd


# Initialize Neo4j driver
@st.cache_resource
def init_driver():
    return GraphDatabase.driver(st.session_state.neo_uri, auth=(st.session_state.neo_user, st.session_state.neo_password))

driver = init_driver()

def fetch_pheno_main_data(driver, phenotype_id):
    query = """
    MATCH (p:phenotype {id: $phenotype_id})
    RETURN p.phenotypes AS name, p.id AS id
    """
    with driver.session() as session:
        result = session.run(query, phenotype_id=phenotype_id)
        return result.single()

def header(phenotype_id):
    main_data = fetch_pheno_main_data(driver, phenotype_id)

    st.markdown(f""" 
        <h1>{main_data["name"]}</h1>
    """)


def render_hyperlink(text):
    # Ensure text is a string before processing
    if isinstance(text, (int, float)):
        text = str(text)
    elif not isinstance(text, str):
        return text

    # Regex to find [name](link) format
    pattern = re.compile(r'\[(.*?)\]\((.*?)\)')
    match = pattern.match(text)
    if match:
        name, link = match.groups()
        return f'<a href="{link}" target="_blank">{name}</a>'
    return text

def display_detail(record_id, tab_name, pid_property):
    # Define the query to find the detail node
    query = f"""
    MATCH (p:phenotype {{id: $phenotype_id}})
    OPTIONAL MATCH (d:{tab_name.lower()}_detail)
    WHERE d.PID = p.{pid_property}
    RETURN d
    """

    with driver.session() as session:
        result = session.run(query, phenotype_id=record_id)
        detail_node = result.single()

    if detail_node and detail_node["d"]:
        # Parse and display the detail node properties
        detail_data = dict(detail_node["d"].items())
        for key, value in detail_data.items():
            # Prepare the key display
            key_display = f"<span style='font-size: 0.8em; font-weight: bold; text-transform: uppercase;'>{key.capitalize()}</span><br>"

            # Initialize the value display
            value_display = ""

            # Check if the value is a list in a string form
            if isinstance(value, str):
                try:
                    # Attempt to parse the string as a list
                    parsed_value = literal_eval(value)
                    if isinstance(parsed_value, list):
                        value = parsed_value
                except (ValueError, SyntaxError):
                    # If parsing fails, keep the original value as a string
                    pass

            if isinstance(value, list):
                # Strip any leading/trailing quotes from each element and create tags
                tags = [render_hyperlink(str(item).strip('\'"')) for item in value]
                value_display = "<div style='display: flex; flex-direction: row; flex-wrap: wrap;'>"
                value_display += ' '.join([f"<span style='font-size: 1.0em; background-color: #2E4052; border-radius: 15px; padding: 4px 8px; margin: 4px 4px 4px 0;'>{tag}</span>" for tag in tags])
                value_display += "</div>"
            else:
                # For non-list values, display as a single item or hyperlink
                value_display = f"<span style='font-size: 1.2em;'>{render_hyperlink(value)}</span>"

            # Combine key and value for display
            st.markdown(f"<div style='margin-bottom: 10px;'>{key_display}{value_display}</div>", unsafe_allow_html=True)
    else:
        st.write(f"No detailed data available for {tab_name}.")

def display_concepts():
    return

# Define the function to get concepts related to the detail node
def get_concepts(record_id, tab_name, pid_property):
    query = f"""
    MATCH (p:phenotype {{id: $phenotype_id}})
    OPTIONAL MATCH (d:{tab_name.lower()}_detail)-[:HAS_CONCEPT]->(c)
    WHERE d.PID = p.{pid_property}
    RETURN d.PID AS detail_pid, c
    """
    with driver.session() as session:
        result = session.run(query, phenotype_id=record_id)

        detail_PID = ""

        concepts = []
        for record in result:
            detail_pid = record["detail_pid"]
            concept_node = record["c"]
            if concept_node:
                # Convert Node object to a dictionary
                concept = dict(concept_node)
                # Iterate over all properties in the concept
                for key, value in concept.items():
                    if isinstance(value, str):
                        try:
                            # Try to evaluate the string as a literal expression
                            evaluated_value = literal_eval(value)
                            concept[key] = evaluated_value
                        except (ValueError, SyntaxError):
                            # If evaluation fails, keep the original string
                            pass
                
                # Handle PIDs and retain array properties based on PIDs
                if "PIDs" in concept and isinstance(concept["PIDs"], list):
                    try:
                        index = concept["PIDs"].index(detail_pid)
                        for key, value in concept.items():
                            if isinstance(value, list) and len(value) > index:
                                concept[key] = value[index]
                    except ValueError:
                        # detail_pid not found in PIDs list, handle accordingly
                        pass
                concepts.append(concept)

    df_concepts = pd.DataFrame(concepts)

    df_concepts = df_concepts.style.format({
        col: lambda x: f"{x:.0f}" for col in df_concepts.select_dtypes(include=['float', 'int']).columns
    })
    
    return df_concepts

def tabs(record_id):
    tab_names = []
    pid_properties = {
        "Sentinel": "sentinel_PID",
        "HDRUK": "hdruk_PID",
        "CPRD": "cprd_PID",
        "OHDSI": "ohdsi_PID",
        "PheKB": "phekb_PID"
    }

    # Determine which tabs to display based on the record ID
    if record_id[0].lower() == 's':
        tab_names.append("Sentinel")
    if record_id[1].lower() == 'h':
        tab_names.append("HDRUK")
    if record_id[2].lower() == 'c':
        tab_names.append("CPRD")
    if record_id[3].lower() == 'o':
        tab_names.append("OHDSI")
    if record_id[4].lower() == 'p':
        tab_names.append("PheKB")

    if tab_names:
        selected_tab = st.tabs(tab_names)
        for i, tab_name in enumerate(tab_names):
            with selected_tab[i]:
                pid_property = pid_properties[tab_name]
                with st.expander(f"## {tab_name} Details"):
                    display_detail(record_id, tab_name, pid_property)
                with st.expander(f"## {tab_name} Concepts"):
                    st.dataframe(get_concepts(record_id, tab_name, pid_property))
    else:
        st.write("No specific data available for this phenotype ID.")
        


def show(phenotype_id):
    phenotype_data = fetch_pheno_main_data(driver, phenotype_id)

    if phenotype_data:
        st.markdown(f"# {phenotype_data['name']}")
        st.markdown(f"#### PID (Phenotype ID): {phenotype_data['id']}")

        tabs(phenotype_id)
    else:
        st.error("No data found for the given phenotype ID.")

# Fetch phenotype_id from query parameters
current_phenotype = st.session_state["current_pheno"]

if current_phenotype:
    show(current_phenotype)
else:
    st.error("No phenotype ID provided.")