from Bio import Entrez
import requests
import sys
import json
import re

LLM_MODEL = "qwen2.5"
LLM_ENDPOINT = "http://localhost:11434/api/"
MAX_ARTICLES = 10
Entrez.email = "your_email@example.com"

def query_pubmed(query):
    """
    Query PubMed database for articles matching the given search term.

    Parameters:
        query (str): The search term used to query PubMed.

    Returns:
        list: A list of PubMed article IDs matching the search term.
    """
    query = re.sub(r'[^a-zA-Z0-9 ]', '', query)
    print("Querying Pubmed...")
    print(query)
    handle = Entrez.esearch(db="pubmed", term=query, retmax=MAX_ARTICLES, sort="relevance")
    record = Entrez.read(handle)
    print(record)
    return record['IdList']

def fetch_articles(ids, rmode="xml", rtype="medline"):
    """
    Fetch articles from PubMed given a list of article IDs.

    Parameters:
        ids (list): A list of PubMed article IDs.
        rmode (str, optional): The return mode ('xml' or 'text'). Default is 'xml'.
        rtype (str, optional): The return type ('medline', 'abstract', etc.).

    Returns:
        dict: A dictionary containing the fetched articles in the specified format.
    """
    print("Fetching articles:")
    print(ids)
    handle = Entrez.efetch(db="pubmed", id=ids, rettype=rtype, retmode=rmode)
    return Entrez.read(handle)

def llm_generate(prompt):
    """
    Generate a response using the chosen LLM model with an API call.

    Parameters:
        prompt (str): The LLM prompt.

    Returns:
        str: A generated response to the prompt.
    """
    url = LLM_ENDPOINT + "generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        #"format": "json",
        "stream": False,
        # "options": {
        #     "num_keep": 5, # number of tokens to keep from previous context
        #     "seed": 42,
        #     "num_predict": 100, # number of tokens to predict
        #     "top_k": 20, # consider k most probable tokens
        #     "top_p": 0.9, # consider tokens with cum probability p
        #     "min_p": 0.0, # filter tokens below probability p
        #     "typical_p": 0.7, # filters unlikely tokens (0-1)
        #     "repeat_last_n": 33, # repetitions of recent tokens
        #     "temperature": 0.8, # randomness of token selection
        #     "repeat_penalty": 1.2, # penalize repeated tokens
        #     "presence_penalty": 1.5, # avoid previously used tokens
        #     "frequency_penalty": 1.0, # avoid repetition in generated text
        #     "mirostat": 1, # enables mirostat to dynmically adjust temperature
        #     "mirostat_tau": 0.8, # target perplexity level for dynamically adjusting temperature
        #     "mirostat_eta": 0.6, # aggressiveness of adjustment
        #     "penalize_newline": True,
        #     "stop": ["\n", "user:"],
        #     "numa": False, # memory adjustment for parallel processing
        #     "num_ctx": 1024, # max tokens in context
        #     "num_batch": 2,
        #     "num_gpu": 1,
        #     "main_gpu": 0,
        #     "low_vram": False,
        #     "vocab_only": False,
        #     "use_mmap": True,
        #     "use_mlock": False,
        #     "num_thread": 8
        # }
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.RequestException as e:
        return f"API request failed: {e}"

def generate_pubmed_query(question):
    """
    Generate a query in Pubmed format from a query in natural language.

    Parameters:
        question (str): A question in natural language.

    Returns:
        str: A query in Pubmed format.
    """
    print("Generating Pubmed query")
    prompt = f"""
        Generate a simple pubmed query based on the following question: {question}
        Output only the generated query with no explanation or any other text.
        Do not use the word 'reasons'. Use 'causes' instead of 'reasons'.

        Examples:
        Input: Does alcohol cause cancer?
        Output: cancer AND alcohol AND causality
        Input: Can diet reduce obesity?
        Output: obesity AND diet AND reduction
        input: What are the most effective treatments for eczema?
        Output: eczema AND treatment AND effectiveness
        input: Why do autistic people stim?
        output: autism AND stimming AND causes
        input: Why are so many people overweight?
        output: overweight AND rising prevalence AND causes
    """
    gen_query = llm_generate(prompt)
    print(f"Generated query: {gen_query}")
    return gen_query

def assess_articles(question, articles):
    """
    Assess a number of scientific articles to answer a question.

    Parameters:
        question (str): The question to be answered.

    Returns:
        str: An answer to the question based on the articles given.
    """
    print("Building abstracts..")
    abstract_string = ""
    for i, article in enumerate(articles):
        # print(article['title'])
        # print(article['abstract'])
        abstract_string += f"""
            Abstract {i}:
            {article['title']}
            {article['abstract']}

        """
    print(abstract_string)
    prompt = f"""
        You are provided with several academic article titles and abstracts. Your task is to synthesize these abstracts and provide a comprehensive 
        summary that captures the key points from all sources.

        Additionally, you will be given a specific question related to one of the articles. Your 
        response should answer this question by referencing information found in the provided 
        abstracts.

        **Input:**

        Abstract 1:
        "Recent policy recommendations have emphasized the importance of renewable energy 
        subsidies to reduce carbon emissions. This study finds that a 10% increase in renewable 
        energy subsidies leads to a 2% decrease in CO2 emissions over five years."

        Abstract 2:
        "Technological advancements in battery storage are crucial for integrating intermittent 
        renewable sources into the grid. The development of high-capacity, low-cost batteries can 
        significantly enhance the reliability and efficiency of renewable energy systems."

        Abstract 3:
        "Economic analysis suggests that investing in public transportation is a cost-effective 
        way to reduce urban carbon footprints. A comprehensive study indicates that cities with 
        robust public transport systems see an average 15% reduction in per capita emissions."

        Abstract 4:
        "Societal behavior plays a critical role in climate change mitigation. This research 
        highlights the importance of community-based initiatives, such as local recycling 
        programs and educational campaigns, which can lead to significant behavioral changes 
        among residents."

        Question: Based on the provided abstracts, what are some key strategies that have been 
        identified for reducing urban carbon footprints?

        **Output:**

        To reduce urban carbon footprints, several key strategies have been identified based on 
        the provided abstracts:

        1. **Renewable Energy Subsidies**: Increasing subsidies for renewable energy can 
        significantly decrease CO2 emissions. A 10% increase in subsidies has been shown to lead 
        to a 2% reduction in CO2 emissions over five years (Abstract 1).

        2. **Technological Advancements**: Developing high-capacity, low-cost batteries is 
        essential for integrating intermittent renewable sources into the grid. These 
        advancements can enhance the reliability and efficiency of renewable energy systems 
        (Abstract 2).

        3. **Public Transportation Investment**: Investing in robust public transportation 
        systems is a cost-effective approach to reducing urban carbon footprints. Cities with 
        strong public transport see an average 15% reduction in per capita emissions (Abstract 
        3).

        4. **Community-Based Initiatives**: Local recycling programs and educational campaigns 
        can foster significant behavioral changes among residents, leading to reduced carbon 
        footprints at the community level (Abstract 4).

        These strategies collectively address different aspects of urban carbon footprint 
        reduction, from policy recommendations to technological innovations, economic analyses, 
        and social behaviors.

        **Input:**
        {abstract_string}
        
        Question: {question}

        **Output:** 

    """
    print("Assessing articles...")
    return llm_generate(prompt)

def run_query(question):
    """
    Run the query from the user-supplied question.

    Parameters:
        question (str): The user-supplied question.

    Returns:
        str: An answer to the question.
    """
    query = generate_pubmed_query(question)
    id_list = query_pubmed(query)
    record = fetch_articles(ids=id_list, rtype="abstract")
    articles = []
    for pa in record['PubmedArticle']:
        try:
            title = pa['MedlineCitation']['Article']['ArticleTitle']
            abstract = pa['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
            articles.append({"title": title, "abstract": abstract})
        except Exception as e:
            print(f"Could not parse article: {e}")
    answer = assess_articles(question, articles)
    return answer


if __name__ == "__main__":
    # Retrieve the search query from command-line arguments or prompt the user
    if len(sys.argv) < 2:
        question = input("Enter search query: ")
    else:
        question = sys.argv[1]
    answer = run_query(question)
    print("\nANSWER\n=================\n")
    print(answer)
