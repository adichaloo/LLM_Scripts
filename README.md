## Sample LLM Scripts
1. LateFusionEnsembling -> We are Ensembling 2 mistral models and fine-tuning on MedMCQA dataset (Biomedical MCQ dataset). The 2 mistral models were trained on 
Medquad QA (Question/Answering dataset) and MedMCQA (AI2’s Reasoning Challenge (ARC) dataset) respectively.

2. Mixture of Experts -> Using again 2 mistral models previously mentioned we train a gating network for selection of a best model for a specific medical task.

3. MedMCQA_Prompts_Data_Preprocessing -> Preprocessing the MedMCQA dataset into prompts which will be used to train our LLMs.

4. baseline_mistral_on_medmcqa -> Finetuning the Mistral 7B model using QLora on a subset of MedMCQA dataset

## Transformer from Scratch
Assignment for developing a single attention head from scratch to predict, for each position in the string, how many times the character at that position occurred previously, maxing out at 2. 
Here a character level tokenization has been used .


