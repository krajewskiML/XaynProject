import json
import matplotlib.pyplot as plt
import seaborn as sns
import re

# READING DATASET TO LIST

all_records = []
with open('news_dataset.json') as json_file:
    for line in json_file:
        record = json.loads(line)
        all_records.append(record)


# CLEARING DATA

def process_text(text):
    text = text.lower().replace('\n', ' ').replace('\r', '').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


for record in all_records:
    for key in record:
        record[key] = process_text(record[key])

# CHECKING DATA SAMPLES
print("First 3 datapoints look like this:")
for record in all_records[:3]:
    print(record['name'])
    print(record['snippet'])
    print(record['topic'])

# DATASET SIZE

max_article_size = max([len((record['snippet']).split()) for record in all_records])
print(f'max snippet size is: {max_article_size}')

snippets_words_by_article = [record['snippet'].split() for record in all_records]
flat_all_snippets_words = [item for sublist in snippets_words_by_article for item in sublist]
name_words_by_article = [record['name'].split() for record in all_records]
flat_all_names_words = [item for sublist in name_words_by_article for item in sublist]

# TOPIC CATEGORIES

topics = [record['topic'] for record in all_records]
topics = set(topics)
print("these are the topics:", topics)
topic_dict = {}
topic_dict_sizes = {}
for topic in topics:
    topic_dict[topic] = [record for record in all_records if record['topic'] == topic]
    topic_dict_sizes[topic] = len(topic_dict[topic])

plt.bar(topic_dict_sizes.keys(), topic_dict_sizes.values(), width=0.2)
plt.title('Count of articles per topic')
plt.xlabel('topics')
plt.ylabel('count')
plt.show()

# WORDS PER ARTICLE DISTRIBUTION

sns.displot([len(word_list) for word_list in snippets_words_by_article]).set_titles('article length distribution')
plt.xlabel('number of words')
plt.ylabel('number of articles')
plt.show()
