# MicrobExpert
Ensure you have Python 3.8+ installed. Install the required dependencies (such as torch, transformers, pandas, scikit-learn):

Step 2: Data Preparation
Prepare your target biomedical literature (e.g., PubMed abstracts or full-text paragraphs). Save your raw textual data into a standard .txt or .csv format so that the NER scripts can read them line by line.

Step 3: Named Entity Recognition (NER)
Navigate to the ner/ folder. Depending on your target domain, open either D-M-NER.ipynb (Disease-Microbe) or F-M-NER.ipynb (Food-Microbe) using Jupyter Notebook.

Run the notebook cells to automatically scan your input texts using our curated dictionaries (dietary_dict.csv, etc.).

Output: The script will output a structured CSV file containing sentences that have co-occurring candidate entities.

Step 4: Relation Extraction via MicrobExpert (RE)
Once candidate sentences are identified, navigate to the MOE/ folder.

Run the MOE.py script, feeding it the output CSV from Step 3.

The MoE model will dynamically route the text through the appropriate expert networks and predict the semantic relationships (Positive, Negative, Relate, or NA).

Step 5: Output and Downstream Application
The final pipeline exports a structured tabular file containing semantic triples (e.g., Food A - Positive - Microbe B) accompanied by their original evidence sentences. You can seamlessly import this output into graph databases (like Neo4j) for network visualization or use it directly for evidence-based hypothesis generation in dietary science.

📊 Data Availability
All curated datasets generated during this study (including Basic GSC, D-M GSC/SSC, and F-M GSC/SSC) are freely available in the data/ directory. Researchers are welcome to use these resources to advance predictive disease modeling and microbiome research.

📝 Citation
If you find our model, code, or datasets useful in your research, please consider citing our paper:

(Citation information will be updated upon publication)
