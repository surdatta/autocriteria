import os
import json
import pandas as pd
import re
from tqdm import tqdm
import logging
import argparse
from bs4 import BeautifulSoup
import glob
from nltk import word_tokenize
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.callbacks import StdOutCallbackHandler

os.environ["OPENAI_API_KEY"] = "<insert_openai_api_key_here>"

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


def generate_time_frame_prompt(sentence_text, attribute_text):
    full_template = """
        {general_instruction}

        {sentence}

        {time_frame}
        """
    full_prompt = PromptTemplate.from_template(full_template)

    general_instruction_template = """Please do not extract anything outside of the given Sentence. The extracted phrase spans should directly be from the given Sentence text. \n\n"""
    general_instruction_prompt = PromptTemplate.from_template(general_instruction_template)

    sentence_template = """[Sentence]: """ + sentence_text + """\n"""
    sentence_prompt = PromptTemplate.from_template(sentence_template)

    time_frame_template = """What is the specific time frame associated with """ + attribute_text + """ in the above sentence? Please be precise (e.g., within 14 days, past 6 months, etc.) If time information is not available, return 'NA'"""
    time_frame_prompt = PromptTemplate.from_template(time_frame_template)

    input_prompts = [
        ("general_instruction", general_instruction_prompt),
        ("sentence", sentence_prompt),
        ("time_frame", time_frame_prompt)
    ]

    time_frame_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

    return time_frame_prompt


def generate_individual_diseases_prompt(sentence_text):
    full_template = """
        {general_instruction}

        {sentence}

        {individual_diseases}
        """
    full_prompt = PromptTemplate.from_template(full_template)

    general_instruction_template = """Please do not extract anything outside of the given Sentence. The extracted phrase spans should directly be from the given Sentence text. \n\n"""
    general_instruction_prompt = PromptTemplate.from_template(general_instruction_template)

    sentence_template = """[Sentence]: """ + sentence_text + """\n"""
    sentence_prompt = PromptTemplate.from_template(sentence_template)

    individual_diseases_template = """What are all the diseases mentioned in this sentence? List each disease in a new line."""
    individual_diseases_prompt = PromptTemplate.from_template(individual_diseases_template)

    input_prompts = [
        ("general_instruction", general_instruction_prompt),
        ("sentence", sentence_prompt),
        ("individual_diseases", individual_diseases_prompt)
    ]

    individual_diseases_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

    return individual_diseases_prompt


def generate_individual_treatments_prompt(sentence_text):
    full_template = """
        {general_instruction}

        {sentence}

        {individual_treatments}
        """
    full_prompt = PromptTemplate.from_template(full_template)

    general_instruction_template = """Please do not extract anything outside of the given Sentence. The extracted phrase spans should directly be from the given Sentence text. \n\n"""
    general_instruction_prompt = PromptTemplate.from_template(general_instruction_template)

    sentence_template = """[Sentence]: """ + sentence_text + """\n"""
    sentence_prompt = PromptTemplate.from_template(sentence_template)

    individual_treatments_template = """What are all the treatment names, therapy names, medication or drug names, and procedure names mentioned in this sentence? List each of them in a new line."""
    individual_treatments_prompt = PromptTemplate.from_template(individual_treatments_template)

    input_prompts = [
        ("general_instruction", general_instruction_prompt),
        ("sentence", sentence_prompt),
        ("individual_treatments", individual_treatments_prompt)
    ]

    individual_diseases_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

    return individual_diseases_prompt


def generate_inclusion_criteria_prompt(criteria_text_inclusion):
    full_template = """
        {general_instruction}

        {inclusion_criteria_text}

        {criteria_of_interest}

        {output_format}

        {criteria_level_instructions}

        {entity_type_classification}

        {entity_type_attribute_relations}
        """
    full_prompt = PromptTemplate.from_template(full_template)

    general_instruction_template = """Please do not extract anything outside of the given Criteria Text. The extracted phrase spans should directly be from the given Criteria Text.\n\n"""
    general_instruction_prompt = PromptTemplate.from_template(general_instruction_template)

    inclusion_criteria_text_template = """[Inclusion Criteria Text]:\n\n""" + criteria_text_inclusion + """\n\n"""
    inclusion_criteria_text_prompt = PromptTemplate.from_template(inclusion_criteria_text_template)

    criteria_of_interest_template = """
    From the above inclusion criteria text, identify the age, gender, education, ' \
    '{disease} or non-alcoholic fatty liver disease (NAFLD) activity score (NAS) requirement, condition for steatosis score, ' \
    'condition for Lobular inflammation score, condition for Ballooning degeneration score, ' \
    'biomarkers (including liver fat content on magnetic resonance imaging proton density fat fraction or MRI-PDFF and fibrosis stage), lab tests, vitals (e.g., BMI), contraception-related criteria, ' \
    'all prior treatments or therapies, medications or drugs, procedures (including imaging scans), {disease} diagnosis criteria, all other diseases or comorbidities, and life expectancy along with any conditions.' \
    'Extract all specific medication and drug names that are used to fight {disease} as well as medications for other diseases. ' \
    'Extract all treatments and therapies including chemotherapy, immunotherapy, targeted therapy, inhibitors, or antibodies against molecules, and interventional studies. ' \
    'Extract all comorbidities including all disease names (both hypernyms and their hyponyms), any health conditions, mental issues, complication, syndromes, disorder, symptoms, abnormalities, allergy, hypersensitivities, contraindication, adverse events, or side effects.\n
    """
    criteria_of_interest_prompt = PromptTemplate.from_template(criteria_of_interest_template)

    response_schemas = [
        ResponseSchema(name="Entity",
                       description="the entity type of an eligibility criteria"),
        ResponseSchema(name="Attribute",
                       description="the text span or phrase associated with an eligibility criteria"),
        ResponseSchema(name="Value",
                       description="the value associated with an eligibility criteria"),
        ResponseSchema(name="Condition",
                       description="any condition/restriction/exception or other details associated with an eligibility criteria"),
        ResponseSchema(name="Sentence",
                       description="the corresponding sentence or phrase in the text where an eligibility criteria was found")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    output_format_template = """
    Show everything in 5 fields with keys Entity, Attribute, Value, Condition, and Sentence.\n{format_instructions}\n' \
    'Include any other condition/restriction/exception, other details, or specific conditions for specific patient groups ' \
    'under the "Condition" field of each treatment or therapy, medication, biomarker, lab test, and disease. ' \
    'Please think deeply to extract this information. ' \
    'Also include the corresponding sentence or phrase in the text where the entity was found under "Sentence" field.' \
    'The phrase under "Attribute" field must be present in the phrase under "Sentence" field. ' \
    'The "Condition" should show temporal/condition information very precisely.' \
    'Show only the lab tests and entries that are mentioned in the above text. ' \
    'Put lab test names under "Attribute" field and their numeric values under "Value" field.' \
    'Only show the lab test names that are mentioned. In case only adequate organ function is mentioned, specify that as a different entity under "Attribute" field.' \
    'Also create different lab test entities when the conditions are different (e.g., varying disease conditions).\n
    """
    output_format_prompt = PromptTemplate.from_template(output_format_template, partial_variables={"format_instructions": format_instructions})

    criteria_level_instructions_template = """
    For Biomarkers, put all gene, gene product names, and imaging or fibrosis biomarkers (including Min_liver_fat MRI-PDFF and fibrosis stage) under "Attribute" field, and their mutation types or expression level or value under "Value" field ' \
                              'and any specific condition under "Condition" field.' \
                              'If mutation type or expression level is not mentioned, put "Yes" for required inclusion, ' \
                              '"Allowed" when they are exceptions and are allowed or eligible under certain specific conditions or time frames, and "No" for not included cases under the "Value" field.' \
                              'Also put the specific condition under the "Condition" field.' \
                              'Put each disease name in the text under "Attribute" field and include either the abbreviated or the full form of the disease.' \
                              'Put any specific condition or the status (e.g., active) described for each disease under "Condition" field.' \
                              'For each disease, previous treatment or therapy, procedure, and medication, put either "Yes" for required inclusion, ' \
                              '"Allowed" when they are exceptions and are allowed or eligible under certain conditions or time frames, or "No" for not included cases under the "Value" field.' \
                              'Also put the specific condition under the "Condition" field.' \
                              'Any tests related to infections/diseases and any imaging exam for any disease should go under "Condition" field.' \
                              'Put the value for Age by including the number (usually with greater than or less than symbols) as is present in the given original criteria text.' \
                              'For steatosis score, Lobular inflammation score, and Ballooning degeneration score, put "Score" under "Entity", put the scoring names under "Attribute" field and put either "1-2" or "1-3" under "Value" if the phrase - "at least 1 in each" is present in the sentence. Also put any condition under the "Condition" field. ' \
                              'For liver fat content on MRI proton density fat fraction, put "Biomarker" under "Entity", put the "Min_liver_fat MRI-PDFF" under "Attribute" field and its value under "Value". ' \
                              'For vitals, put the vital category under "Attribute" and the value (e.g., weight change percentages, BMI values, etc.) under "Value" field. ' \
                              'For non-alcoholic steatohepatitis (NASH) diagnosis criteria, put "Diagnosis" under "Entity" field, put the disease name under "Attribute" and put either "Yes" for required inclusion, "Allowed" for allowed criteria, or "No" for not included cases under "Value". ' \
                              'Put each entity in each row. ' \
                              'Each unique disease entity must be in different rows.\n
    """
    criteria_level_instructions_prompt = PromptTemplate.from_template(criteria_level_instructions_template)

    entity_type_classification_template = """
    Classify each entity into one of the following classes - Demographic, Vital, Score, Contraceptive, Biomarker, Diagnosis, Comorbidity, Previous Treatment, Lab test, and Survival.' \
                              'And put that class under the "Entity" field.' \
                              'Every row should have unique phrase under "Attribute" field.' \
                              'Entity field can contain Demographic, Vital, Score, Contraceptive, Biomarker, Diagnosis, Comorbidity, Previous Treatment, Lab test, and Survival.\n
    """
    entity_type_classification_prompt = PromptTemplate.from_template(entity_type_classification_template)

    entity_type_attribute_relations_template = """
    Note that the "Comorbidity" class includes all disease terms, whereas the "Previous Treatment" class includes all treatments (including antibody or inhibitor treatments), medications, therapies, drugs, and procedures. ' \
                              'Attribute field will include "Age" for age criteria, "Gender" for gender criteria, and "Education" for education criteria. ' \
                              '"Age", "Gender", and "Education" falls under "Demographic" entity, and "NASH score", "steatosis score", "Lobular inflammation score", and "Ballooning degeneration score" under "Score" entity, and "Min_liver_fat MRI-PDFF" and "Fibrosis stage" under "Biomarker" entity.' \
                              '"Attribute" field will include "Life Expectancy" for the life expectancy criteria. If it is mentioned in the text, put the life expectancy time period under the "Value" field, otherwise put "NA".' \
                              '"Life Expectancy" falls under "Survival" entity.\n
    """
    entity_type_attribute_relations_prompt = PromptTemplate.from_template(entity_type_attribute_relations_template)

    input_prompts = [
        ("general_instruction", general_instruction_prompt),
        ("inclusion_criteria_text", inclusion_criteria_text_prompt),
        ("criteria_of_interest", criteria_of_interest_prompt),
        ("output_format", output_format_prompt),
        ("criteria_level_instructions", criteria_level_instructions_prompt),
        ("entity_type_classification", entity_type_classification_prompt),
        ("entity_type_attribute_relations", entity_type_attribute_relations_prompt)
    ]

    in_criteria_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

    return in_criteria_prompt, output_parser



def generate_exclusion_criteria_prompt(criteria_text_exclusion):
    full_template = """
        {general_instruction}

        {exclusion_criteria_text}

        {criteria_of_interest}

        {output_format}

        {criteria_level_instructions}

        {entity_type_classification}

        {entity_type_attribute_relations}
        """
    full_prompt = PromptTemplate.from_template(full_template)

    general_instruction_template = """Please do not extract anything outside of the given Criteria Text. The extracted phrase spans should directly be from the given Criteria Text.\n\n"""
    general_instruction_prompt = PromptTemplate.from_template(general_instruction_template)

    exclusion_criteria_text_template = """[Exclusion Criteria Text]:\n\n""" + criteria_text_exclusion + """\n\n"""
    exclusion_criteria_text_prompt = PromptTemplate.from_template(exclusion_criteria_text_template)

    criteria_of_interest_template = """
    From the above exclusion criteria text, extract all prior treatments or therapies, medications or drugs, procedures (including imaging scans), ' \
                              'lab tests, biomarkers, vitals (e.g., BMI), ' \
                              '{disease} diagnosis criteria, ' \
                              'all other diseases or comorbidities, life expectancy, and ' \
                              'contraception-related criteria.' \
                              'Extract all specific medication and drug names that are used to fight {disease} as well as medications for other diseases. ' \
                              'Extract all treatments and therapies including chemotherapy, immunotherapy, targeted therapy, inhibitors, or antibodies against molecules, and interventional studies. ' \
                              'Extract all comorbidities including all disease names (both hypernyms and their hyponyms), any health conditions, mental issues, complication, syndromes, disorder, symptoms, abnormalities, allergy, hypersensitivities, contraindication, adverse events, or side effects.\n
    """
    criteria_of_interest_prompt = PromptTemplate.from_template(criteria_of_interest_template)

    response_schemas = [
        ResponseSchema(name="Entity",
                       description="the entity type of an eligibility criteria"),
        ResponseSchema(name="Attribute",
                       description="the text span or phrase associated with an eligibility criteria"),
        ResponseSchema(name="Value",
                       description="the value associated with an eligibility criteria"),
        ResponseSchema(name="Condition",
                       description="any condition/restriction/exception or other details associated with an eligibility criteria"),
        ResponseSchema(name="Sentence",
                       description="the corresponding sentence or phrase in the text where an eligibility criteria was found")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    output_format_template = """
    Show everything in 5 fields with keys Entity, Attribute, Value, Condition, and Sentence.\n{format_instructions}\n' \
                              'Include any condition/restriction/exception, other details, or specific conditions for specific patient groups ' \
                              'under the "Condition" field of each treatment, biomarker, disease, procedure, and medication. ' \
                              'Please think deeply to extract this information. ' \
                              'Also include the corresponding sentence or phrase in the text where the entity was found under "Sentence" field.' \
                              'The phrase under "Attribute" field must be present in the phrase under "Sentence" field. ' \
                              'The "Condition" should show temporal/condition information very precisely.\n
    """
    output_format_prompt = PromptTemplate.from_template(output_format_template, partial_variables={"format_instructions": format_instructions})

    criteria_level_instructions_template = """
    For Biomarkers, put all gene, gene product names, and receptor names under "Attribute" field, and their mutation types or expression level or value under "Value" field ' \
                              'and any specific condition under "Condition" field.' \
                              'If mutation type or expression level is not mentioned, the value field is either "Yes" or "Allowed". ' \
                              'The Value is "Yes" for required exclusion. However, the Value is "Allowed" when they are exceptions and are allowed or eligible under certain specific conditions or time frames. ' \
                              'Also put the specific condition under the "Condition" field.' \
                              'Put each disease name in the text under "Attribute" field and include either the abbreviated or the full form of the disease.' \
                              'Put any specific condition or the status (e.g., active) described for each disease under "Condition" field.' \
                              'For each disease, treatment or therapy, procedure, and medication, the value field is either "Yes" or "Allowed". ' \
                              'The Value is "Yes" for required exclusion. However, the Value is "Allowed" when they are exceptions and are allowed or eligible under certain specific conditions or time frames. ' \
                              'Also put the specific condition under the "Condition" field.' \
                              'Any tests related to infections/diseases and any imaging exam for any disease should go under "Condition" field.' \
                              'For vitals, put the vital category under "Attribute" and the value (e.g., weight change percentages, BMI values, etc.) under "Value" field. ' \
                              'Put each entity in each row. ' \
                              'Each unique disease entity must be in different rows. ' \
                              'Every row should have unique phrase under "Attribute" field.\n
    """
    criteria_level_instructions_prompt = PromptTemplate.from_template(criteria_level_instructions_template)

    entity_type_classification_template = """
    Classify each entity into one of the following classes - Biomarker, Vital, Diagnosis, Comorbidity, Contraceptive, Previous Treatment, Lab test, and Survival. And put that class under the "Entity" field. The "Entity" field must contain one class from this list - Biomarker, Vital, Diagnosis, Comorbidity, Contraceptive, Previous Treatment, 
    Lab test, and Survival.\n
    """
    entity_type_classification_prompt = PromptTemplate.from_template(entity_type_classification_template)

    entity_type_attribute_relations_template = """
    Note that the "Comorbidity" class includes all disease terms, whereas the "Previous Treatment" class includes all treatments (including antibody or inhibitor treatments), medications, therapies, drugs, and procedures. ' \
                              '"Attribute" field will include "Life Expectancy" for the life expectancy criteria. If it is mentioned in the text, put the life expectancy time period under the "Value" field, otherwise put "NA".' \
                              '"Life Expectancy" falls under "Survival" entity.\n
    """
    entity_type_attribute_relations_prompt = PromptTemplate.from_template(entity_type_attribute_relations_template)

    input_prompts = [
        ("general_instruction", general_instruction_prompt),
        ("exclusion_criteria_text", exclusion_criteria_text_prompt),
        ("criteria_of_interest", criteria_of_interest_prompt),
        ("output_format", output_format_prompt),
        ("criteria_level_instructions", criteria_level_instructions_prompt),
        ("entity_type_classification", entity_type_classification_prompt),
        ("entity_type_attribute_relations", entity_type_attribute_relations_prompt)
    ]

    ex_criteria_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

    return ex_criteria_prompt, output_parser


def generate_prompts(criteria_text_ex, criteria_text_in):
    prompts = {}
    prompt_in, o_p_in = generate_inclusion_criteria_prompt(criteria_text_in)
    prompt_ex, o_p_ex = generate_exclusion_criteria_prompt(criteria_text_ex)

    prompts['in'] = {'prompt': prompt_in, 'criteria_text': criteria_text_in, 'output_parser': o_p_in}
    prompts['ex'] = {'prompt': prompt_ex, 'criteria_text': criteria_text_ex, 'output_parser': o_p_ex}
    return prompts


def process(message, df_output, type, trial_ID, no_retries, d_time, max_toks, phase_str, lnk_str, cri_text_sent):
 
    for extracted_criteria in message:
        entity = extracted_criteria['Entity'].strip()
        attribute = extracted_criteria['Attribute'].strip()
        value = extracted_criteria['Value'].strip()
        condition = extracted_criteria['Condition'].strip()
        sentence = extracted_criteria['Sentence'].strip()


        not_found_string = ['-', '', 'na', 'n/a', 'not mentioned', 'not specified', 'not available']

        # Does not contain source sentence information
        if sentence.lower() in not_found_string or 'from the above exclusion criteria text, extract the' in sentence.lower() or 'from the above inclusion criteria text, identify the' in sentence.lower():
            continue

        if attribute.lower() in not_found_string and value.lower() in not_found_string:
            logging.warning('Both attribute and value columns are null.')
            logging.info('Trial ID: {} -- Criteria: {} -- LINE: {}\n'.format(trial_ID, type, line))
            continue

        if 'no diseases are mentioned' in attribute.lower() or 'no diseases mentioned' in attribute.lower() or 'no specific diseases mentioned' in attribute.lower() or 'no specific diseases are mentioned' in attribute.lower():
            continue

        if condition.lower() in not_found_string:
            condition = 'NA'

        try:
            assert attribute.lower() not in not_found_string and entity.lower() not in not_found_string
        except AssertionError:
            logging.warning('Both entity and attribute columns are null.')
            logging.info('Trial ID: {} -- Criteria: {} -- LINE: {}\n'.format(trial_ID, type, line))
            continue

        cr_single_value = ['age', 'gender', 'score']

        if any(cr in attribute.lower() for cr in cr_single_value) and value.lower() in not_found_string:
            continue
        
        cr_ent_value = ['score', 'performance score', 'diagnosis']
        if any(cr in entity.lower() for cr in cr_ent_value) and value.lower() in not_found_string:
            continue

        
        # ignoring attributes with name in them such as Gene Name and Disease Name
        if 'name' in attribute.lower():
            continue

        # may get information from prompt
        if 'mutation type' in value.lower() or 'numeric value' in value.lower():
            continue
        # Any other diseases
        list_attribute_terms_to_avoid = ['other disease', 'other diseases', 'procedure', 'procedures', 'comorbidity', 'comorbidities', 'medication', 'medications',
                                            'contraception-related criteria', 'treatments or therapies', 'treatment or therapy', 'treatments', 'therapies', 'drugs', 'treatment', 'therapy', 'drug', 'contraception', 'diagnosis', 'diagnoses',
                                            'other diseases or comorbidities', 'other disease or comorbidity', 'prior therapy', 'prior treatment', 'prior therapies', 'prior treatments']
        if any(attr in attribute.lower() for attr in list_attribute_terms_to_avoid) or attribute.lower() == 'biomarker':
            continue

        # for cases when time frame conditions go to value column
        time_indicators = ['week', 'month', 'year', 'day', 'cycle']
        if any(ti in value.lower() for ti in time_indicators) and condition.lower() in ['na', 'n/a']:
            condition = value
            value = 'Yes'

        # for cases like life expectancy when the time period goes to condition and value is yes/na
        if any(ti in condition.lower() for ti in time_indicators) and value.lower() in ['na','yes'] and entity.lower() == 'survival' and attribute.lower() == 'life expectancy':
            value = condition
            condition = 'NA'

        if condition.lower() in not_found_string and value.lower() in ['na',
                                                                        'yes'] and entity.lower() == 'survival' and attribute.lower() == 'life expectancy':
            continue

        # Handling cases like - Biomarker | EFGR | mutation | NA
        # For cases like - Biomarker | EGFR | Mutation | No -- condition will be checked
        # and if condition is Yes/No the value will be updated accordingly
        # Biomarker | KRAS | Mutations | Not excluded
        # For other cases, the value will be changed to Yes
        if 'mutation' == value.lower() or 'rearrangement' == value.lower() or 'mutations' == value.lower() \
                or 'rearrangements' == value.lower() or 'alteration' == value.lower() or 'alterations' == value.lower():
            attribute = attribute + ' ' + value
            if condition.lower() in ['yes', 'no', 'not included', 'not excluded', 'not eligible', 'not enrolled']:
                if 'not' in condition.lower():
                    value = 'No'
                    condition = 'NA'
                else:
                    value = condition
                    condition = 'NA'
            else:
                value = 'Yes'

        # Cases when Biomarker protein names are shown but haven't captured antibody/inhibitor information from the sentence
        if entity.lower() == 'biomarker' and ('antibod' in sentence.lower() or 'inhibitor' in sentence.lower()):
            entity = 'Previous Treatment'
            if 'antibod' in sentence.lower():
                attribute = attribute + ' ' + 'antibody'
            elif 'inhibitor' in sentence.lower():
                attribute = attribute + ' ' + 'inhibitor'

        # Sometimes the Values are null but Conditions have Yes/No
        if attribute.lower() not in not_found_string and value.lower() in not_found_string and condition.lower() in ['yes', 'no']:
            value = condition
            condition = 'NA'

        elif attribute.lower() not in not_found_string and value.lower() in not_found_string and condition.lower() not in ['yes', 'no']:
            # Sometimes lab test values are not captured/not available
            if entity in ['Lab Test', 'Procedure']:
                value = 'NA'
            else:
                value = 'Yes'

        # Sometimes the values are 'Yes/No'
        if value.lower() == 'yes/no' or value.lower() == 'no/yes':
            value = 'Yes'

        # Change Procedure and Previous Treatment to Treatment History
        if entity.lower() in ['procedure', 'previous treatment']:
            entity = 'Treatment History'

        # sometimes the age value goes to modifier column
        if attribute.lower() in ['age'] and any(char_con.isdigit() for char_con in condition):
            value = condition

        # Separate multiple diseases in different rows
        words_indicative_of_multiple_diseases = ['including', 'such as', 'examples', 'include', 'like', 'or',
                                                    'and']

        time_frame_response = 'NA'
        # the following entity types generally do not have any time frame constraints
        if entity.lower() not in ['demographic', 'performance score', 'tumor characteristics', 'score']:

            prompt_time_frame = generate_time_frame_prompt(sentence, attribute)

            o_p = None

            time_frame_response = generate_response_for_partial_doc(prompt_time_frame, o_p, no_retries, d_time, max_toks, trial_ID, type)
            # print('Time frame response: \n')
            # print(time_frame_response)
            
            if time_frame_response is not None:
                time_frame_response = time_frame_response.strip()
                if time_frame_response.endswith('.'):
                    time_frame_response = time_frame_response[:-1]

                not_available_time = ['na', 'n/a', 'not mentioned', 'not specified', 'not available', 'not found', 'not applicable']
                if any(nf_str in time_frame_response.lower() for nf_str in not_available_time):
                    time_frame_response = 'NA'

        # Comorbidity -- multiple diseases extraction in a sentence
        
        if entity.lower() == 'comorbidity':
            prompt_for_individual_diseases = generate_individual_diseases_prompt(sentence)

            original_com_attribute = attribute.strip()

            o_p = None
            diseases_response = generate_response_for_partial_doc(prompt_for_individual_diseases, o_p, no_retries, d_time, max_toks, trial_ID, type)

            # print('Diseases response: \n')
            # print(diseases_response)
            
            if diseases_response is not None:
                diseases = diseases_response.split('\n')
                for disease in diseases:
                    disease = disease.strip()

                    if disease != '' and '---' not in disease and ((disease not in original_com_attribute)
                                                                    and (original_com_attribute not in disease)):
                        attribute_com = disease
                        if condition.lower().strip().count(' ') == 0 and condition.lower().strip() != 'na' and condition.lower().strip() != 'allowed':
                            first_word_disease = disease.split(' ')[0].lower().strip()
                            if first_word_disease != condition.lower().strip():
                                attribute_com = condition + ' ' + disease

                        val_ind_disease = value

                        row = {'Trial ID': trial_ID, 'Type': type, 'Phase': phase_str, 'URL': lnk_str, 'Entity': entity,
                                'Attribute': attribute_com, 'Value': val_ind_disease, 'Temporal': time_frame_response,
                                'Modifier': condition,
                                'Source Sentence': sentence}

                        df_output = pd.concat([df_output, pd.DataFrame([row])])
                        # if in-situ cancers are present, then 'previous malignancies' are also present
                        if 'in-situ' in attribute_com:
                            previous_malignancies_row = {'Trial ID': trial_ID, 'Type': type, 'Phase': phase_str, 'URL': lnk_str,
                                                            'Entity': entity,
                                                            'Attribute': 'Previous malignancies', 'Value': 'Yes',
                                                            'Temporal': time_frame_response,
                                                            'Modifier': 'NA',
                                                            'Source Sentence': sentence}
                            df_output = pd.concat([df_output, pd.DataFrame([previous_malignancies_row])])

        # Check if entity is comorbidity and the modifier contains only word,
        # then prepend the attribute phrase with the modifier
        if entity.lower() == 'comorbidity' and condition.lower().strip().count(
                ' ') == 0 and condition.lower().strip() != 'na' and condition.lower().strip() != 'allowed':
            # first word of original attribute is not matching the modifier word
            attribute = attribute.strip()
            first_word_attribute = attribute.split(' ')[0].lower().strip()
            if first_word_attribute != condition.lower().strip():
                attribute = condition + ' ' + attribute

        
        if entity.lower() == 'treatment history':

            prompt_for_individual_treatments = generate_individual_treatments_prompt(sentence)

            original_tr_attribute = attribute.strip()

            o_p = None
            treatment_response = generate_response_for_partial_doc(prompt_for_individual_treatments, o_p, no_retries, d_time, max_toks, trial_ID, type)
            # print('Treatment response: \n')
            # print(treatment_response)
            
            if treatment_response is not None:
                treatments = treatment_response.split('\n')
                for treatment in treatments:
                    treatment = treatment.strip()

                    if treatment != '' and '---' not in treatment and ((treatment not in original_tr_attribute)
                                                                        and (original_tr_attribute not in treatment)):
                        attribute_ind_tr = treatment
                        val_ind_tr = value

                        row = {'Trial ID': trial_ID, 'Type': type, 'Phase': phase_str, 'URL': lnk_str, 'Entity': entity,
                                'Attribute': attribute_ind_tr, 'Value': val_ind_tr, 'Temporal': time_frame_response,
                                'Modifier': condition,
                                'Source Sentence': sentence}

                        df_output = pd.concat([df_output, pd.DataFrame([row])])

        
        row = {'Trial ID': trial_ID, 'Type': type, 'Phase': phase_str, 'URL': lnk_str,
               'Entity': entity,
               'Attribute': attribute, 'Value': value, 'Temporal': time_frame_response,
               'Modifier': condition,
               'Source Sentence': sentence}
        df_output = pd.concat([df_output, pd.DataFrame([row])])

    return df_output


def generate_response(prompt, n_retry, delay, max_tokens, tr_id, cri_type):
    if cri_type == 'in':
        cri_type = 'Inclusion'
    else:
        cri_type = 'Exclusion'
    for i in range(n_retry):
        try:
            llm = ChatOpenAI(model_name="gpt-4", temperature=0)
            chain = LLMChain(llm=llm, prompt=prompt['prompt'])
            # printing the prompt text
            # handler = StdOutCallbackHandler()
            # chain = LLMChain(llm=llm, prompt=prompt['prompt'], callbacks=[handler])
            output = chain.run({"disease": " non-alcoholic steatohepatitis (NASH)"})
            out_parser = prompt['output_parser']
            # output = out_parser.parse(output)
            json_strings = output.split("```json\n")[1:]
            list_of_dicts = [json.loads(s.split("\n```")[0]) for s in json_strings]

            # print("JSON Output: \n")
            # print(list_of_dicts)
            return list_of_dicts
        except Exception as e:
            logging.error(str(e))
            logging.error('Trial ID: {} -- Criteria: {} -- ERROR: Other Exception!'.format(tr_id, cri_type))
    logging.error('Trial ID: {} -- Criteria: {} -- ERROR: Failed to generate text after '
                  '{} retries'.format(tr_id, cri_type, str(n_retry)))


def generate_response_for_partial_doc(prompt, output_p, n_retry, delay, max_tokens, tr_id, cri_type):
    if cri_type == 'in':
        cri_type = 'Inclusion'
    else:
        cri_type = 'Exclusion'
    for i in range(n_retry):
        try:
            llm = ChatOpenAI(model_name="gpt-4", temperature=0)
            chain = LLMChain(llm=llm, prompt=prompt)
            # printing the prompt text
            # handler = StdOutCallbackHandler()
            # chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])

            output = chain.run({"disease": " non-alcoholic steatohepatitis (NASH)"})
            if output_p is not None:
                # output = output_p.parse(output)
                json_strings = output.split("```json\n")[1:]
                output = [json.loads(s.split("\n```")[0]) for s in json_strings]

            # print("JSON Output: \n")
            # print(output)
            return output
        except Exception as e:
            logging.error(str(e))
            logging.error('Trial ID: {} -- Criteria: {} -- ERROR: Other Exception!'.format(tr_id, cri_type))
    logging.error('Trial ID: {} -- Criteria: {} -- ERROR: Failed to generate text after '
                  '{} retries'.format(tr_id, cri_type, str(n_retry)))
        


def divide_document(text):
    # Split the text into sentences
    sentences = text.split('. ')

    # Find the index of the sentence that is closest to half the length of the text
    half_length = len(text) // 2
    index = 0
    length = 0
    for i, sentence in enumerate(sentences):
        length += len(sentence) + 2
        if length > half_length:
            index = i
            break

    # Join the sentences up to the chosen index to create the first half
    first_half = '. '.join(sentences[:index + 1])

    # Join the remaining sentences to create the second half
    second_half = '. '.join(sentences[index + 1:])

    return first_half, second_half


def process_half_text_response(half_text_response, df_half_output, cri_type, ph_str, link_str, tr_id, num_retry, delay_t, m_toks, criteria_text_sent):
    if cri_type == 'in':
        cri_type = 'Inclusion'
    else:
        cri_type = 'Exclusion'
    if half_text_response == 'INPUT STILL LONG EVEN AFTER SPLIT OR CHUNK TOO LONG':
        logging.error('Trial ID: {} -- Criteria: {} -- ERROR: Input text is still long even '
                      'after splitting the trial text or chunk is too long'.format(tr_id, cri_type))
        return df_half_output
    elif half_text_response is not None:
        return process(half_text_response, df_half_output, cri_type, tr_id, num_retry, delay_t, m_toks, ph_str, link_str, criteria_text_sent
                       )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Running GPT on clinical trial documents to extract eligibility criteria.')
    parser.add_argument('-input_file', '--input_xml_file', help='File path to the directory of xml files containing raw clinical trial data')
    parser.add_argument('-output_file', '--output_file_extracted_entities',
                        help='File path to an excel file storing all extracted criteria with each sheet containing criteria for a trial document')
    parser.add_argument('-log_file', '--log_file_path',
                        help='File path to log file containing all information, warning and error messages')

    args = parser.parse_args()

    log_file_path = args.log_file_path
    input_file_path = args.input_xml_file

    max_tokens = 2000
    num_retries = 3
    delay_time = 600
    chunk_size = 200

    trial_index = 0

    output_file_path = args.output_file_extracted_entities
    logging.basicConfig(level=logging.INFO, filename=log_file_path,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    files = glob.glob(f'{input_file_path}/*.xml')
    # files = files[0:1]
    for filename in tqdm(files):
        with open(filename, 'r') as f:
            trial_data = f.read()

            df_write = pd.DataFrame(columns=['Trial ID', 'Type', 'Phase', 'URL', 'Entity',
                                             'Attribute', 'Value', 'Temporal', 'Modifier', 'Source Sentence'
                                             ])
            trial_data = BeautifulSoup(trial_data, "xml")

            trial_id = trial_data.find('nct_id').text
            trial_id = str(trial_id)
            ex_criteria_text = None
            in_criteria_text = None

            seq_num = int(trial_index) + 1
            logging.info('Processing sequence number: {}, with trial ID: {}\n'.format(seq_num, trial_id))

            min_age_str = None
            max_age_str = None
            gender_str = None
            if trial_data.find('minimum_age'):
                min_age_str = trial_data.find('minimum_age').text
            if trial_data.find('maximum_age'):
                max_age_str = trial_data.find('maximum_age').text
            if trial_data.find('gender'):
                gender_str = trial_data.find('gender').text

            all_eligibility = trial_data.find_all('eligibility')
            for eligibility in all_eligibility:
                criteria = eligibility.find('criteria')
                criteria_text_block = criteria.find('textblock').text

                
                if 'inclusion criteria' in criteria_text_block.lower() and 'exclusion criteria' in criteria_text_block.lower():
                    splits_by_inclusion = re.split('Inclusion Criteria|Inclusion criteria|inclusion criteria|inclusion Criteria|INCLUSION CRITERIA', criteria_text_block, 1)
                    text_following_inclusion = splits_by_inclusion[1]
                    splits_by_exclusion = re.split('Exclusion Criteria|Exclusion criteria|exclusion criteria|exclusion Criteria|EXCLUSION CRITERIA',
                                                   text_following_inclusion, 1)
                    in_criteria_text = splits_by_exclusion[0].strip()

                    # print('''===Inclusion===\n''')
                    # print(in_criteria_text)

                    ex_criteria_text = splits_by_exclusion[1].strip()
                    # print('''===Exclusion===''')
                    # print(ex_criteria_text)

            if in_criteria_text is None and ex_criteria_text is None:
                logging.warning('Trial ID: {} -- Either inclusion or exclusion criteria text is null.'.format(trial_id))
                continue

            phase_str = trial_data.find('phase').text

            if '/' in phase_str:
                f_p = phase_str.split('/')[0]
                s_p = phase_str.split('/')[1]
                f_p = f_p.lower().split('phase')[1].strip()
                s_p = s_p.lower().split('phase')[1].strip()
                phase_str = f_p + '/' + s_p
            else:
                phase_str = phase_str.lower().split('phase')[1].strip()

            url_str = trial_data.find('url').text

            prompts_dict = generate_prompts(ex_criteria_text, in_criteria_text)

            for item, prompt in prompts_dict.items():

                criteria_text_length = len(word_tokenize(prompt['criteria_text']))
                if criteria_text_length <= chunk_size:
                    print('Criteria text: {} of trial ID {} contains less than {} words'.format(item, trial_id, chunk_size))

                    response_text = generate_response(prompt, num_retries, delay_time, max_tokens, trial_id, item)

                    if response_text == 'INPUT TOO LONG':
                        # split criteria text into halves
                        if item == 'in':
                            first, second = divide_document(in_criteria_text)
                            reduced_prompt_first_half, o_p_first = generate_inclusion_criteria_prompt(first)
                            reduced_prompt_second_half, o_p_second = generate_inclusion_criteria_prompt(second)

                        else:
                            first, second = divide_document(ex_criteria_text)

                            reduced_prompt_first_half, o_p_first = generate_exclusion_criteria_prompt(first)
                            reduced_prompt_second_half, o_p_second = generate_exclusion_criteria_prompt(second)


                        response_text_first = generate_response_for_partial_doc(reduced_prompt_first_half, o_p_first, num_retries,
                                                                                delay_time, max_tokens, trial_id, item)
                        response_text_second = generate_response_for_partial_doc(reduced_prompt_second_half, o_p_second,
                                                                                 num_retries, delay_time, max_tokens,
                                                                                 trial_id, item)
                        df_write = process_half_text_response(response_text_first, df_write, item, phase_str, url_str,
                                                              trial_id, num_retries, delay_time, max_tokens, prompt['criteria_text'])
                        df_write = process_half_text_response(response_text_second, df_write, item, phase_str, url_str,
                                                              trial_id, num_retries, delay_time, max_tokens, prompt['criteria_text'])

                    elif response_text is not None:
                        if item == 'in':
                            df_write = process(response_text, df_write, 'Inclusion', trial_id, num_retries, delay_time,
                                               max_tokens, phase_str, url_str, prompt['criteria_text'])
                        else:
                            df_write = process(response_text, df_write, 'Exclusion', trial_id, num_retries, delay_time,
                                               max_tokens, phase_str, url_str, prompt['criteria_text'])
                else:
                    print('Criteria text: {} of trial ID {} contains more than {} words'.format(item, trial_id, chunk_size))

                    # Dividing the criteria text into chunks preserving sentence boundaries
                    chunks = []
                    chunk = ''
                    words = 0
                    # for sentence in re.split(r'(\. |\? |\! |\n)', prompt['criteria_text']):
                    for sentence in re.split(r'(\. |\? )', prompt['criteria_text']):
                        # print('Sentence: \n')
                        # print(sentence)
                        if len(sentence.strip()) > 0:
                            sentence_words = len(word_tokenize(sentence))
                            if words + sentence_words <= chunk_size:
                                chunk += sentence + ' '
                                words += sentence_words
                            else:
                                chunks.append(chunk)
                                chunk = sentence + ' '
                                words = sentence_words
                    if len(chunk) > 0:
                        chunks.append(chunk)

                    for chunk in chunks:
                        # print('Chunk:\n')
                        # print(chunk)

                        # Chunks that only contain a one or two-digit number followed by a period.
                        pattern = r"^\d{1,2}\.$"
                        match = re.match(pattern, chunk)
                        if match:
                            continue

                        if item == 'in':
                            reduced_prompt_chunk, o_p = generate_inclusion_criteria_prompt(chunk)
                        else:
                            reduced_prompt_chunk, o_p = generate_exclusion_criteria_prompt(chunk)

                        response_text_chunk = generate_response_for_partial_doc(reduced_prompt_chunk, o_p, num_retries,
                                                                                delay_time, max_tokens, trial_id, item)
                        # print("Response: \n")
                        # print(response_text_chunk)
                        df_write = process_half_text_response(response_text_chunk, df_write, item, phase_str, url_str,
                                                              trial_id, num_retries, delay_time, max_tokens, chunk)

            df_write['Attribute'] = df_write['Attribute'].str.replace(r'(^(-|\s-))', '', regex=True)
            df_write['Attribute'] = df_write['Attribute'].str.replace(r'(^(?:[1-9]|[1-9][0-9]|100)\. ?)', '', regex=True)
            df_write['Attribute'] = df_write['Attribute'].str.strip()
            df_write['Attribute_Lowercase'] = df_write['Attribute'].str.lower()


            def is_substring(row):
                text = row['Attribute'].lower()
                for other_text in df_write['Attribute'].str.lower():
                    # here we apply the length of text constraint to avoid removing short attributes such as ast and alt which can occur as a substring in other attribute phrases
                    if text in other_text and text != other_text and len(text) > 4:
                        return True
                return False


            # to select the longest overlapping attribute phrase
            df_write = df_write[~df_write.apply(is_substring, axis=1)]
            
            # dropping duplicate attribute phrases for Comorbidity, Lab Test, and Treatment History entities
            df_write_com_lab = df_write[df_write['Entity'].isin(['Comorbidity', 'Lab Test', 'Treatment History'])].copy()
            df_write_com_lab_dedup = df_write_com_lab.drop_duplicates(subset=['Attribute_Lowercase'], keep='first')
            df_write_others = df_write[~df_write['Entity'].isin(['Comorbidity', 'Lab Test', 'Treatment History'])].copy()
            component_dfs = [df_write_com_lab_dedup, df_write_others]
            df_write_com_lab_dedup_plus_others_all = pd.concat(component_dfs)
            df_write = df_write_com_lab_dedup_plus_others_all.drop(columns=['Attribute_Lowercase'])


            for i in range(len(df_write)):
                row = df_write.iloc[i]

                # adding in-situ to allowed excepted cancers
                # can make it more restrictive
                cancer_related_words = ['cancer', 'carcinoma', 'dysplasia']
                # if row['Entity'].lower().strip() == 'comorbidity' and any(c in row['Attribute'].lower().strip() for c in
                #                                                          cancer_related_words) and row['Value'].lower().strip() == 'allowed':
                # rule for excepted cancers
                if row['Entity'].lower().strip() == 'comorbidity' and any(c in row['Attribute'].lower().strip() for c in cancer_related_words) and 'meningitis' not in row['Attribute'].lower():
                    corresponding_sent = row['Source Sentence'].lower().strip()
                    original_attr = row['Attribute'].strip()
                    for word in ['except', 'excluding', 'permitted', 'allowed', 'still eligible', 'may be eligible']:
                        if word in corresponding_sent:
                            index = corresponding_sent.index(word) + len(word) + 1  # index of the word after the matched word
                            if any(c in corresponding_sent[index:] for c in cancer_related_words):
                                row['Attribute'] = original_attr + ', (in-situ)'
                                row['Value'] = 'Allowed'
                                break

            # remove "Prior LOT" attributes, any attribute phrase containing "Any", and any attribute containing none-like values
            df_write = df_write[~df_write['Attribute'].isin(['Prior LOT', 'Any'])]
            search_for_none_values = ['none', 'n/a', 'not mentioned', 'not specified', 'not available', 'not found', 'not applicable']
            df_write = df_write[~df_write['Attribute'].str.contains('|'.join(search_for_none_values), case=False)]

            
            df_write = df_write[
                ~df_write['Attribute'].str.lower().isin(
                    ['no diseases are mentioned in this sentence.', 'no diseases mentioned in this sentence.',
                     'no specific diseases mentioned', 'no specific diseases are mentioned',
                     'there are no diseases mentioned in this sentence.',
                     'there are no specific diseases mentioned in this sentence.',
                     'there are no specific treatment names, therapy names, medication or drug names, or procedure names mentioned in this sentence.',
                     'there are no treatment names, therapy names, medication or drug names, or procedure names mentioned in this sentence.',
                     'there are no specific disease or health condition terms mentioned in this sentence.',
                     'there are no disease or health condition terms mentioned in this sentence.',
                     'there are no disease, health condition, or comorbidity terms mentioned in this sentence.',
                     'there are no disease or health condition terms mentioned in the given sentence.'])]

            
            df_write = df_write[
                ~df_write['Attribute'].str.lower().isin(
                    ['other disease', 'other diseases', 'procedure', 'procedures', 'comorbidity', 'comorbidities',
                     'medication', 'medications',
                     'contraception-related criteria', 'treatments or therapies', 'treatment or therapy', 'treatments',
                     'therapies', 'drugs', 'treatment', 'therapy', 'drug', 'contraception', 'diagnosis', 'diagnoses',
                     'other diseases or comorbidities', 'other disease or comorbidity', 'prior therapy',
                     'prior treatment', 'prior therapies', 'prior treatments', 'gene', 'gene product', 'receptor',
                     'hormone receptor', 'imaging biomarker',
                     'genes', 'gene products', 'receptors', 'hormone receptors', 'imaging biomarkers', 'imaging scan',
                     'imaging scans', 'complication', 'complications', 'issues', 'issue',
                     'side effects', 'syndromes', 'adverse events', 'side effect', 'syndrome', 'adverse event',
                     'mental issue', 'mental issues', 'allergy', 'disorder', 'symptoms', 'abnormalities', 'symptom', 'disorders'])]

            # Populating available values for age and gender
            if min_age_str is not None and max_age_str is not None:
                age_str = '>=' + min_age_str + ' and ' + '<=' + max_age_str
                df_write.loc[df_write['Attribute'] == 'Age', 'Value'] = age_str
            elif min_age_str is None and max_age_str is not None:
                age_str = '<=' + max_age_str
                df_write.loc[df_write['Attribute'] == 'Age', 'Value'] = age_str
            elif min_age_str is not None and max_age_str is None:
                age_str = '>=' + min_age_str
            else:
                age_str = 'NA'

            if gender_str is None:
                gender_str = 'NA'

            if not df_write['Attribute'].isin(['Age']).any():
                new_df = pd.DataFrame(
                    {'Trial ID': [trial_id], 'Type': ['Inclusion'], 'Phase': [phase_str], 'URL': [url_str],
                     'Entity': ['Demographic'],
                     'Attribute': ['Age'], 'Value': [age_str], 'Temporal': ['NA'],
                     'Modifier': ['NA'], 'Source Sentence': ['Extracted from structured field.']})
                df_write = pd.concat([df_write, new_df], ignore_index=True)
            else:
                missing_values = ['na', 'n/a', 'not available', 'none', 'not specified', 'not found', '-', '']
                age_rows = df_write[df_write['Attribute'] == 'Age']
                if not age_rows.empty and age_rows['Value'].isin(missing_values).any():
                    df_write.loc[df_write['Attribute'] == 'Age', 'Value'] = age_str

            if not df_write['Attribute'].isin(['Gender']).any():
                new_df = pd.DataFrame(
                    {'Trial ID': [trial_id], 'Type': ['Inclusion'], 'Phase': [phase_str], 'URL': [url_str],
                     'Entity': ['Demographic'],
                     'Attribute': ['Gender'], 'Value': [gender_str], 'Temporal': ['NA'],
                     'Modifier': ['NA'], 'Source Sentence': ['Extracted from structured field.']})
                df_write = pd.concat([df_write, new_df], ignore_index=True)
            else:
                missing_values = ['na', 'n/a', 'not available', 'none', 'not specified', 'not found', '-', '']
                gender_rows = df_write[df_write['Attribute'] == 'Gender']
                if not gender_rows.empty and gender_rows['Value'].isin(missing_values).any():
                    df_write.loc[df_write['Attribute'] == 'Gender', 'Value'] = gender_str

            df_write = df_write.sort_values('Type', ascending=False)

            with pd.ExcelWriter(output_file_path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
                df_write.to_excel(writer, sheet_name=trial_id)
