import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter

# Ensure stopwords and punctuation are downloaded and set up
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
# Adding common but non-informative words specific to medications and treatments
additional_stopwords = {'used', 'use', 'treat', 'treatment', 'including', 'common', 'drugs', 'may', 'cause', 'causes', 'help', 'also', 'works', 'side', 'effects', 'effect'}
stop_words.update(additional_stopwords)

def scrape_drugs():
    base_url = "https://www.drugs.com/drug_information.html"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    drug_list_section = soup.find('ul', {'class': 'ddc-list-column-4'})
    if not drug_list_section:
        print("Drug list section not found.")
        return pd.DataFrame()
    
    drug_links = [link['href'] for link in drug_list_section.find_all('a', href=True)[:50]]
    drugs_data = []
    for link in drug_links:
        drug_url = f"https://www.drugs.com{link}"
        drug_info = scrape_drug_info(drug_url)
        drugs_data.append(drug_info)
        time.sleep(1)
    
    return pd.DataFrame(drugs_data)

def scrape_drug_info(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    drug_name = soup.find('h1').text.strip() if soup.find('h1') else 'N/A'
    description_tag = soup.find('h2', string=lambda text: text and 'What is' in text)
    description = description_tag.find_next('p').text.strip() if description_tag and description_tag.find_next('p') else 'N/A'
    
    side_effects = []
    side_effects_tag = soup.find('h2', id='side-effects')
    if side_effects_tag:
        element = side_effects_tag.find_next_sibling()
        while element:
            if element.name == 'h2':
                break
            side_effects.append(element.text.strip())
            element = element.find_next_sibling()
        side_effects = ' '.join(side_effects).replace('\n', ' ')
    else:
        side_effects = 'N/A'

    return {'Drug Name': drug_name, 'Uses': description, 'Side Effects': side_effects}

def clean_data(df):
    df['Drug Name'] = df['Drug Name'].str.replace('[^a-zA-Z0-9\s]', '').str.strip()
    df['Uses'] = df['Uses'].str.lower()
    df['Side Effects'] = df['Side Effects'].str.lower()
    return df.drop_duplicates(subset=['Drug Name'])

def analyze_data(df):
    unique_drug_count = df['Drug Name'].nunique()
    print(f"Number of Unique Drug Names: {unique_drug_count}")
    
    # Analyzing common uses with filtering punctuations and stopwords
    uses_counter = Counter()
    for text in df['Uses'].dropna():
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
        bigrams = [' '.join(gram) for gram in ngrams(filtered_tokens, 2)]
        uses_counter.update(bigrams)
    print("Top 5 Most Common Uses:", uses_counter.most_common(5))
    
    # Analyzing side effects with improved filtering
    side_effects_counter = Counter()
    for text in df['Side Effects'].dropna():
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
        side_effects_counter.update(filtered_tokens)
    print("Most Common Side Effect:", side_effects_counter.most_common(1))

def main():
    df_raw = scrape_drugs()
    if not df_raw.empty:
        df_clean = clean_data(df_raw)
        analyze_data(df_clean)
        df_clean.to_csv('cleaned_drug_data.csv', index=False)
        print("Data scraping, cleaning, and analysis complete. Cleaned data saved to 'cleaned_drug_data.csv'.")
    else:
        print("No drug data was scraped.")

if __name__ == "__main__":
    main()
