# AutoCriteria: A Generalizable Clinical Trial Eligibility Criteria Extraction System Powered by Large Language Models

We leverage the GPT-4 model to extract granular eligibility criteria information from clinical trial documents (collected from 
https://ClinicalTrials.gov/) covering a variery of diseases (e.g., oncology, Alzheimer’s, rare diseases).


## Main steps

1. Prepare data -- Download the xml files containing the clinical trial eligibility criteria text from ClinicalTrials.gov and store them in a directory
2. Run `eligibility_criteria_extraction.py` to extract all criteria (including contextual information such as temporality and conditions) corresponding to all the trial documents collected in Step 1
Sample command (tested using python 3):
```
python eligibility_criteria_extraction.py -input_file <path_to_directory_containing_trial_xml_files> -output_file <path_to_excel_file_to_store_output> -log_file <path_to_log_file>
```