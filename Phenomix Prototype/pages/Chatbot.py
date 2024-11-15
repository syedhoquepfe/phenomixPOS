import streamlit as st
from openai import OpenAI
import os

from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate

from neo4j import GraphDatabase

import json

node_properties_relationships = """ 

PID and CID are global ID system that spans across the databasaes. Listed below are the node labels and their properties:

        {
            phenotype: a master list of all the phenotypes across all the databases (Sentinel, HDRUK, CPRD, OHDSI, PheKb)
            phenotype_properties: {
                "phenotypes": name of the phenotype,
                "sentinel_PID": associated Sentinel PID
                "hdruk_PID": string of an array of associated HDRUK PIDs,
                "cprd_PID": associated CPRD PID,
                "ohdsi_PID": associated OHDSI PID,
                "phekb_PID": associated PheKb PID
            }

            sentinel_detail: details of the phenotype in the Sentinel database,
            sentinel_detail_properties: {
                "outcome": string, the specific outcome/phenotype being analyzed,
                "algorithm_to_define_outcome": string, description of how codes were used to define outcome/phenotype,
                "description": string, detailed information and reference link for the algorithm,
                "PID": string, unique identifier for the Sentinel phenotype,
                "query_end_date": date,
                "title": string, title of the study or analysis used to compule codes,
                "request_send_date": date, date when the data request was sent,
                "request_id": array of strings, identifiers for the data request of the report,
                "query_start_date": date, start date for the data query period
            }

            sentinel_concept: concept used to define Sentinel phenotype/outcome,
            sentinel_concept_properties: {
                "code": string, medical code associated with the concept,
                "code_type": array of strings, types of medical codes like CPT-4 and RE,
                "code_category": array of strings, category of the medical code like 'Procedure' or 'Diagnosis',
                "principal_diagnosis": array of strings, primary diagnosis information,
                "description": string, name of the concept,
                "request_id": array of arrays of strings, identifiers for the data request,
                "PIDs": array of strings, unique identifiers for the sentinel_details its associated with,
                "outcome": array of strings, the specific outcomes/phenotypes being analyzed in gathering the codes,
                "CID": string, unique identifier for the Sentinel concept,
                "care_setting": array of strings, types of care settings,
            }

            cprd_detail: details of the phenotype in the CPRD database,
            cprd_detail_properties: {
                "disease": string, name of the disease/phenotype,
                "PID": string, unique identifier for the CPRD phenotype,
                "disease_num": integer, numerical identifier for the disease unique to CPRD
            }

            cprd_concept: concept used to define CPRD phenotype/disease,
            cprd_concept_properties: {
                "system_num": array of integers, system number associated with the concept,
                "mapping": array of strings, type of code mapping used,
                "disease": array of strings, name of the disease that the code is associated with,
                "medcode": array of floats, medical code associated with the concept,
                "PIDs": array of strings, unique identifiers for the cprd_details its associated with,
                "snomedctconceptid": array of floats, SNOMED CT concept identifier,
                "medcodeid": array of floats, medical code identifier,
                "descr": string, name of the concept,
                "system": array of strings, system classification,
                "read_code": string, Read code associated with the concept,
                "snomedctdescriptionid": array of floats, SNOMED CT description identifier,
                "disease_num": array of integers, numerical identifier for the disease,
                "category": array of strings, category of the diagnosis,
                "CID": string, unique identifier for the CPRD concept
            }

            hdruk_detail: details of the phenotype in the HDRUK database,
            hdruk_detail_properties: {
                "owner": string, owner of the dataset,
                "event_date_end": date, end date of the event,
                "created": datetime, creation timestamp of the record,
                "author": string, authors of the study,
                "sex": array of strings, sex of the subjects,
                "PID": string, unique identifier for the HDRUK phenotype,
                "event_date_start": date, start date of the event,
                "world_access": integer, access level for the world,
                "type": array of strings, type of disease or syndrome,
                "data_sources": array of strings, sources of the data,
                "group_access": integer, access level for the group,
                "coding_system": array of strings, coding system used,
                "collections": array of strings, collections the dataset belongs to,
                "name": string, name of the phenotype,
                "phenotype_version_id": integer, version identifier for the phenotype,
                "phenotype_id": string, identifier for the report,
                "definition": string, definition of the phenotype,
                "updated": datetime, timestamp of the last update,
                "status": integer, status of the phenotype,
                "publications": array of dictionaries, publications related to the phenotype
            }

            ohdsi_detail: details of the phenotype in the OHDSI database,
            ohdsi_detail_properties: {
                "criteriaLocationVisitSourceConceptPrimaryCriteria": integer, primary criteria for visit source concept,
                "addedDate": date, date when the record was added,
                "hasWashoutInText": integer, indicator if washout period is mentioned in the text,
                "domainConditionOccurrence": integer, indicator for condition occurrence domain,
                "contributorOrcIds": array of strings, ORCID IDs of the contributors,
                "updatedDate": date, date when the record was last updated,
                "domainDeviceExposure": integer, indicator for device exposure domain,
                "criteriaLocationConditionSourceConceptInclusionRules": integer, inclusion rules for condition source concept,
                "criteriaLocationConditionSourceConceptPrimaryCriteria": integer, primary criteria for condition source concept,
                "numberOfInclusionRules": integer, number of inclusion rules,
                "domainVisitOccurrence": integer, indicator for visit occurrence domain,
                "isCirceJson": boolean, indicator if the data is in Circe JSON format,
                "censorWindowStartDate": integer, start date for the censor window,
                "criteriaLocationProviderSpecialtyInclusionRules": integer, inclusion rules for provider specialty,
                "initialEventRestrictionAdditionalCriteriaLimit": string, limit for additional criteria on initial event restriction,
                "demographicCriteriaGender": integer, indicator for gender demographic criteria,
                "domainProcedureOccurrence": integer, indicator for procedure occurrence domain,
                "isReferenceCohort": boolean, indicator if the cohort is a reference cohort,
                "censorWindowEndDate": integer, end date for the censor window,
                "criteriaLocationProcedureSourceConceptPrimaryCriteria": integer, primary criteria for procedure source concept,
                "domainObservation": integer, indicator for observation domain,
                "continousObservationWindowPrior": integer, prior continuous observation window,
                "domainDeath": integer, indicator for death domain,
                "domainMeasurement": integer, indicator for measurement domain,
                "inclusionRuleQualifyingEventLimit": string, limit for qualifying event in inclusion rules,
                "exitDateOffSet": float, offset for exit date,
                "collapseSettingsType": string, type of collapse settings,
                "exitDateOffSetField": string, field for exit date offset,
                "numberOfCohortEntryEvents": integer, number of cohort entry events,
                "criteriaLocationAgePrimaryCriteria": integer, primary criteria for age,
                "criteriaLocationGenderAdditionalCriteria": integer, additional criteria for gender,
                "criteriaLocationMeasurementSourceConceptPrimaryCriteria": integer, primary criteria for measurement source concept,
                "modifiedDate": date, date when the record was last modified,
                "criteriaLocationVisitTypePrimaryCriteria": integer, primary criteria for visit type,
                "numberOfDomainsInEntryEvents": integer, number of domains in entry events,
                "domainDrugEra": integer, indicator for drug era domain,
                "criteriaLocationAgeInclusionRules": integer, inclusion rules for age,
                "restrictedByVisit": boolean, indicator if restricted by visit,
                "status": string, current status of the record,
                "hashTag": array of strings, hashtags associated with the record,
                "cohortId": integer, unique identifier for the cohort within the OHDSI database,
                "contributorOrganizations": array of strings, organizations of the contributors,
                "initialEventRestrictionAdditionalCriteria": boolean, additional criteria for initial event restriction,
                "demographicCriteriaAge": integer, indicator for age demographic criteria,
                "useOfObservationPeriodInclusionRule": integer, use of observation period in inclusion rule,
                "logicDescription": string, description of the logic,
                "cohortName": string, name of the cohort/phenotype,
                "criteriaLocationGenderInclusionRules": integer, inclusion rules for gender,
                "criteriaLocationFirstInclusionRules": integer, first inclusion rules,
                "criteriaLocationGenderPrimaryCriteria": integer, primary criteria for gender,
                "collapseEraPad": integer, era padding for collapse,
                "criteriaLocationProviderSpecialtyPrimaryCriteria": integer, primary criteria for provider specialty,
                "cohortNameFormatted": string, formatted name of the cohort/phenotype,
                "continousObservationWindowPost": integer, post continuous observation window,
                "exitStrategy": string, strategy for exit,
                "eventCohort": integer, event cohort,
                "lastModifiedBy": integer, ID of the last modifier,
                "initialEventLimit": string, limit for initial event,
                "librarian": string, librarian's email,
                "PID": string, unique identifier for the OHDSI phenotype,
                "domainsInEntryEvents": array of strings, domains in entry events,
                "domainObservationPeriod": integer, indicator for observation period domain,
                "demographicCriteria": integer, indicator for demographic criteria,
                "criteriaLocationAgeAdditionalCriteria": integer, additional criteria for age,
                "peerReviewerOrcIds": integer, ORCID IDs of the peer reviewers,
                "ohdsiForumPost": string, link to the OHDSI forum post,
                "createdDate": date, date when the record was created,
                "exitSurveillanceWindow": integer, surveillance window for exit,
                "recommendedReferentConceptIds": array of strings, recommended referent concept IDs,
                "contributors": array of strings, names of the contributors,
                "numberOfConceptSets": integer, number of concept sets,
                "criteriaLocationFirstPrimaryCriteria": integer, first primary criteria,
                "domainDrugExposure": integer, indicator for drug exposure domain
            }

            ohdsi_concept: concept used to define OHDSI cohort/phenotype
            ohdsi_concept_properties: {
                "cohortName": array of strings, names of the associated cohorts/phenotypes,
                "cohortId": array of integers, identifiers for the associated cohorts/phenotypes,
                "ConceptId": integer, unique identifier for the concept within the OHDSI database,
                "ConceptName": string, name of the concept,
                "VocabularyId": array of strings, vocabulary identifiers,
                "ConceptCode": array of strings, codes associated with the concept,
                "ok": boolean, status flag,
                "PIDs": array of strings, unique identifiers for the ohdsi_details its associated with,
                "CID": string, unique identifier for the concept
            }

            phekb_detail: detail of the phenotype in the PheKB database,
            phekb_detail_properties: {
                "date_created": date, date when the record was created,
                "description": string, detailed description of the phenotype,
                "PID": string, unique identifier for the PheKB phenotype,
                "genders": array of strings, genders applicable to the phenotype,
                "phenotype_attributes": array of strings, attributes associated with the phenotype,
                "type_of_phenotype": string, type of the phenotype,
                "races": array of strings, races applicable to the phenotype,
                "name": string, name of the phenotype,
                "phenotype_id": integer, unique identifier for the phenotype within the PheKB database,
                "files": array of strings, URLs to related files,
                "ages": array of strings, age groups applicable to the phenotype,
                "status": string, current status of the phenotype,
                "authors": array of strings, authors of the study
            }

            phekb_concept: concept used to define PheKB phenotype,
            phekb_concept_properties: {
                "name": string, name of the associated phenotype,
                "phenotype_id": integer, unique identifier for the phenotype,
                "files": array of strings, URLs to related files that define phenotype and associated concepts,
                "phenotype_attributes": array of strings, attributes associated with the phenotype,
                "PIDs": array of strings, unique identifiers for the phekb_details its associated with,
                "CID": string, unique identifier for the concept
            }
        }



        The relationships are:

        (:phenotype)-[:DETAILS_ARE]->(:sentinel_detail),
        (:phenotype)-[:DETAILS_ARE]->(:cprd_detail),
        (:phenotype)-[:DETAILS_ARE]->(:hdruk_detail),
        (:phenotype)-[:DETAILS_ARE]->(:ohdsi_detail),
        (:phenotype)-[:DETAILS_ARE]->(:phekb_detail),
        (:sentinel_detail)-[:HAS_CONCEPT]->(:sentinel_concept),
        (:cprd_detail)-[:HAS_CONCEPT]->(:cprd_concept),
        (:ohdsi_detail)-[:HAS_CONCEPT]->(:ohdsi_concept),
        (:phekb_detail)-[:HAS_CONCEPT]->(:phekb_concept)

        Note that hdruk_detail does NOT have a relationship with hdruk_concept and hdruk_concept doesn't exist. All concept/code related questions should exclude hdruk.


"""

def init_driver():
    return GraphDatabase.driver(st.session_state.neo_uri, auth=(st.session_state.neo_user, st.session_state.neo_password))

def get_schema():

    driver = init_driver()
    
    with driver.session(database="neo4j") as session:
        result = session.run("CALL apoc.meta.schema() YIELD value RETURN value").data()

    session.close()
    driver.close()

    return result


def get_properties():

    driver = init_driver()
    detail_arr = ["cprd_detail", "hdruk_detail", "ohdsi_detail", "phekb_detail", "sentinel_detail"]

    node_property = {}

    with driver.session(database="neo4j") as session:
        for detail in detail_arr:

            result = session.run(f"MATCH (n:{detail}) WITH n, keys(n) AS keys UNWIND keys AS key RETURN DISTINCT key").data()
            node_property[detail] = result
    
    return node_property


def nl_to_cypher(question):
    
    #schema = get_schema()
    #properties = get_properties()

    assistant = f""" 

        You are a cypher generation expert. Given a question, use neo4j database schema and the properties of each node type
        to generate cypher that will best answer the question. Only return the cypher and no other text. When necessary, feel free to 
        generate complex cypher queries, like ones that use the APOC plugin. 
        
        {node_properties_relationships}

        Return your response in the following format: ```cypher```

        Note that the following warnings are common; avoid them by all means: If you're using UNION, alias the names to match. for example 
        
        MATCH(cc:CreditCard)
        RETURN cc
        UNION
        MATCH(dc:DebitCard)
        RETURN dc

        must be

        MATCH(cc:CreditCard)
        RETURN cc AS Card
        UNION
        MATCH(dc:DebitCard)
        RETURN dc AS Card

        Take a close look at the properties you're using. If they have spaces in between them, account for them. 


    """

    prompt = question

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "system", "content": assistant},
            {"role": "user", "content": prompt}
        ]
    )

    result = response.choices[0].message.content

    return result

def cypher_to_answer(question, cypher):

    cypher = cypher.strip("```")
    schema = get_schema()
    properties = get_properties()

    driver = init_driver()
    with driver.session(database="neo4j") as session:
        cypher_result = session.run(cypher).data()
    

    assistant = f""" 
    
    You are a cypher interpretation expert. The cypher, {cypher}, has been generated given a question about a phenotype database. Running
    the cypher returns {cypher_result}. Answer the question concisely such that you are speaking to an epidemiologist using the relevant cypher results.
    If not necessary or explicitly requested by the question, do not return any database logistical information like id. 
    Do not include any outside information.

    {node_properties_relationships}

    """

    prompt = question

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "system", "content": assistant},
            {"role": "user", "content": prompt}
        ]
    )

    result = response.choices[0].message.content

    return result


def all_phenotype():
    driver = init_driver()
    query = """
    MATCH (p:phenotype)
    RETURN p.id AS id, p.phenotypes AS name, 
           p.ohdsi_PID AS ohdsi_PID, 
           p.sentinel_PID AS sentinel_PID, 
           p.hdruk_PID AS hdruk_PID,
           p.cprd_PID AS cprd_PID, 
           p.phekb_PID AS phekb_PID
    """
    
    with driver.session(database="neo4j") as session:
        results = session.run(query).data()


    driver.close()

    return results


def find_phenotype(text):

    driver = init_driver()
    results = []
    processed_ids = set()

    for pheno in all_pheno:
        if pheno['id'] not in processed_ids and pheno['name'].lower() in text.lower():
            processed_ids.add(pheno['id'])
            details = {
                'sentinel_detail': [],
                'cprd_detail': [],
                'hdruk_detail': [],
                'ohdsi_detail': [],
                'phekb_detail': []
            }
            
            if pheno['sentinel_PID']:
                pid = pheno['sentinel_PID']
                query = f"MATCH (d:sentinel_detail {{PID: '{pid}'}}) RETURN d"
                with driver.session(database="neo4j") as session:
                    result = session.run(query).data()
                if result:
                    details['sentinel_detail'].extend(result)

            if pheno['cprd_PID']:
                pid = pheno['cprd_PID']
                query = f"MATCH (d:cprd_detail {{PID: '{pid}'}}) RETURN d"
                with driver.session(database="neo4j") as session:
                    result = session.run(query).data()
                if result:
                    details['cprd_detail'].extend(result)

            if pheno['hdruk_PID']:
                try:
                    pids = eval(pheno['hdruk_PID'])
                    if isinstance(pids, list):
                        for pid in pids:
                            query = f"MATCH (d:hdruk_detail {{PID: '{pid}'}}) RETURN d"
                            with driver.session(database="neo4j") as session:
                                result = session.run(query).data()
                            if result:
                                details['hdruk_detail'].extend(result)
                except (SyntaxError, NameError):
                    pass

            if pheno['ohdsi_PID']:
                pid = pheno['ohdsi_PID']
                query = f"MATCH (d:ohdsi_detail {{PID: '{pid}'}}) RETURN d"
                with driver.session(database="neo4j") as session:
                    result = session.run(query).data()
                if result:
                    details['ohdsi_detail'].extend(result)

            if pheno['phekb_PID']:
                pid = pheno['phekb_PID']
                query = f"MATCH (d:phekb_detail {{PID: '{pid}'}}) RETURN d"
                with driver.session(database="neo4j") as session:
                    result = session.run(query).data()
                if result:
                    details['phekb_detail'].extend(result)

            results.append({
                'name': pheno['name'],
                'id': pheno['id'],
                'details': details
            })

    driver.close()
    return results

def pheno_desc(text):

    pheno_info = find_phenotype(text)
    
    assistant = """ 

        You are a phenotype database summarization expert. Phenotypes are sourced from the Sentinel, CPRD, HDRUK, PheKB, and OHDSI databases. 
        Given a raw description of a phenotype, summarize the description from each database to be used in research protocols 
        in a RFC 8259 compliant JSON in the following format:

        {
         name: string,
         sentinel_summary: string,
         cprd_summary: string,
         hdruk_summary: string,
         ohdsi_summary: string,
         phekb_summary: string
        }

        If there is no detail specified for a specific database, leave the string empty. Do not return anything else but the JSON. 
    """

    descs = []

    for pheno in pheno_info:

        prompt = str(pheno)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages = [
                {"role": "system", "content": assistant},
                {"role": "user", "content": prompt}
            ]
        )

        result = response.choices[0].message.content
        descs.append(result)
    
    return descs


def display_desc(text):

    raw_data = pheno_desc(text)
    for entry in raw_data:
        data = json.loads(entry)
        name = data.get('name', 'Unknown')
        sentinel_summary = data.get('sentinel_summary', 'No data available')
        cprd_summary = data.get('cprd_summary', 'No data available')
        hdruk_summary = data.get('hdruk_summary', 'No data available')
        ohdsi_summary = data.get('ohdsi_summary', 'No data available')
        phekb_summary = data.get('phekb_summary', 'No data available')
        
        with st.expander(name):
            st.markdown(f"### Sentinel\n{sentinel_summary}")
            st.markdown(f"### CPRD\n{cprd_summary}")
            st.markdown(f"### HDRUK\n{hdruk_summary}")
            st.markdown(f"### OHDSI\n{ohdsi_summary}")
            st.markdown(f"### PheKB\n{phekb_summary}")


def related_pheno(text):
    driver = init_driver()
    related_pheno = []

    # Extract raw phenotype descriptions
    raw_data = pheno_desc(text)
    
    # Parse the raw data and extract names
    pheno_names = []
    for entry in raw_data:
        if entry.strip():  # Ensure the string is not empty
            try:
                data = json.loads(entry)
                name = data.get('name', 'Unknown')
                if name != 'Unknown':
                    pheno_names.append(name)
            except json.JSONDecodeError:
                # Handle or log the error if needed
                continue  # Skip entries that can't be parsed

    # Add extracted names to related_pheno list
    related_pheno.extend(pheno_names)
    
    # Store concepts related to these phenotypes
    concepts = set()

    # Query to find concepts related to these phenotypes
    for name in pheno_names:
        query = f"""
        MATCH (p:phenotype)-[:HAS_CONCEPT]->(c)
        WHERE p.name = '{name}'
        RETURN c
        """

        with driver.session(database="neo4j") as session:
            result = session.run(query).data()

        for record in result:
            # Ensure the record and necessary fields are present
            if 'c' in record and 'CID' in record['c']:
                concepts.add(record['c']['CID'])

    # Find other phenotypes sharing these concepts
    if concepts:
        concept_ids = ', '.join(f"'{concept}'" for concept in concepts)

        query = f"""
        MATCH (p:phenotype)-[:HAS_CONCEPT]->(c)
        WHERE c.CID IN [{concept_ids}]
        RETURN p.name AS name, COUNT(c) AS shared_count
        ORDER BY shared_count DESC
        """

        with driver.session(database="neo4j") as session:
            result = session.run(query).data()
        
        # Add related phenotypes to the list, excluding those already in pheno_names
        for record in result:
            if record['name'] not in pheno_names:
                related_pheno.append(record['name'])
    
    driver.close()
    return related_pheno


def display_related(text):

    related = related_pheno(text)

    box_style = """
    <style>
    .container {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
    }
    .rounded-box {
        border-radius: 20px;
        padding: 4px 8px;
        margin: 2px 5px;
        color: white;
        background-color: darkblue;
    }
    </style>
    """
    
    st.markdown(box_style, unsafe_allow_html=True)
    html_content = f'<div class="container">'
    for entity in related:
        html_content += f'<div class="rounded-box">{entity}</div>'
    html_content += '</div>'

    st.markdown(html_content, unsafe_allow_html=True)

def show():

    cypher_switch = st.checkbox("Cypher")
    desc_switch = st.checkbox("Relevant Phenotype Description")
    related_switch = st.checkbox("Related Phenotypes")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a question about the database"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        #try:

        cypher = nl_to_cypher(prompt)

        if cypher_switch:
            st.write("**Cypher:**\n")
            st.write(f"{cypher}\n\n")

        answer = cypher_to_answer(prompt, cypher)
        st.write("**Answer:**\n")
        st.write(f"{answer}\n\n")

        #except Exception as e:
        #  msg = f"An error occurred: {str(e)}"
        
        st.session_state.messages.append({"role": "assistant", "content": msg})
        with st.chat_message("assistant"):
            st.write(msg)

            if desc_switch:
                st.write("**Relevant Phenotype Descriptions:**\n")
                display_desc(answer)
            
            if related_switch:
                st.write("**Related phenotypes:**\n")
                display_related(answer)

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title('Chatbot')

# Initialize the OpenAI client
api_key = os.getenv('OPENAI_API_KEY2')
client = OpenAI(api_key = api_key)

# Initialize phenotype IDs
all_pheno = all_phenotype()
show()
