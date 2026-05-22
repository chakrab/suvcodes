# Resume Sorter
This project will use multiple agents to do the following:

 1. Preprocess all Resumes
    a. Generate Markup
    b. Extract Key fields
    c. Store as CSV 

 1. Get a Requirement
 2. Find 10 best candidates fitting the requirement by parsing resume
    a. Find the best fit directory name where candidate availability may be there
    b. Process resumes using pymupdf4llm
    c. Send each resume to an agent to calculate fitness score
    d. Send this dataset to a different agent to get 10 candidates
 3. Send an Email Invite to each candidate

Resume:
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/data