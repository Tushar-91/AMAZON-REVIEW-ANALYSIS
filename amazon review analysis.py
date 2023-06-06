#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',None)
pd.set_option('display.max_colwidth',None)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string

import warnings
warnings.filterwarnings('ignore')


# ## IMPORTING DATA

# In[ ]:


df = pd.read_csv('amazon.csv')


# In[ ]:


df.head()


# ## CHECKING NULL VALUES

# In[ ]:


df.isna().sum()


# In[ ]:


df = df.dropna(axis=0,how='any')


# In[ ]:


df.duplicated().sum()


# ## DATA ENGINEERING

# In[ ]:


df.columns


# ##### cleaning the columns

# In[ ]:


df['discounted_price'] = df['discounted_price'].replace('[^0-9]','',regex = True).astype(int)


# In[ ]:


df['actual_price'] = df['actual_price'].replace('[^0-9.]','',regex = True).astype(float).astype(int)


# In[ ]:


df['rating_count'] = df['rating_count'].round().astype(int)


# In[ ]:


df['discount_percentage'] = df['discount_percentage'].str.rstrip('%').astype(int)


# In[ ]:


df['rating'] = df['rating'].replace('|','')


# In[ ]:


df.drop(df[df['product_id']=='B08L12N5H1'].index,inplace = True)


# In[ ]:


df['rating'] = df['rating'].astype(float)


# In[ ]:


df['user_id'] = range(1,len(df)+1)
df.insert(12,'user_id',df.pop('user_id'))


# ## Assigning a code to each unique category for making visuals easy to understand

# In[ ]:


df['category_code'] = df['category'].astype('category').cat.codes
df.insert(2,'category_code',df.pop('category_code'))


# ## Added a new column to the DataFrame called after_discount which calculates the amount saved after applying a discount to the original price

# In[ ]:


df['after_discount'] = df['actual_price'] - df['discounted_price']
df.insert(4,'after_discount',df.pop('after_discount'))


# In[ ]:


nltk.download('stopwords')
nltk.download('snowball_data')


# ### TEXT PROCESSING ON COLUMNS PRODUCT NAME, PRODUCT_DESCRIPTION AND REVIEWS
# 
# ### CREATED A SNOWBALL STEMMER OBJECT FOR ENGLISH LANGUAGE
# 
# ### CREATED A SET OF STOPWORDS FOR ENGLISH LANGUAGE
# 
# ### DEFINED A FUNCTION CALLED CLEAN_TEXT() THAT TAKES A STRING OF TEXT AS INPUT AND PERFORMED THE FOLLOWING OPERATIONS:
# 
# ### CONVERTING THE TEXT TO LOWERCASE

# In[ ]:


# create snowball stemmer
stemmer = SnowballStemmer('english')

# create set of stopwords
stopwords_set = set(stopwords.words('english'))


# In[ ]:


# function for cleaning text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('','',string.punctuation))
    text = text.translate(str.maketrans('','',string.digits))
    text = [stemmer.stem(word) for word in text.split() if word not in stopwords_set]
    text = ' '.join(text)    
    return text


# In[ ]:


df['product_name'] = df['product_name'].apply(clean_text)
df['about_product'] = df['about_product'].apply(clean_text)
df['review_title'] = df['review_title'].apply(clean_text)


# In[ ]:


print(df['review_title'].iloc[0])


# ## SUMMARY OF DATA ENGINEERING
# 
# ### 1.CLEANED THE DATA BY REMOVING NULL VALUES AND ENSURING CONSISTENCY IN THE DATA
# 
# ### 2.CHANGED THE DATA TYPES OF COLUMNS WHERE REQUIRED TO ENSURE APPROPRIATE DATA MANIPULATION
# 
# ### 3.ADDED A NEW COLUMN CALLED 'AFTER DISCOUNT' WHICH CALCULATES THE AMOUNT SAVED AFTER APPLYING A DISCOUNT TO THE ORIGINAL PRICE
# 
# ### 4.PREPROCESSED THE TEXT DATA IN THE REVIEWS, PRODUCT NAME AND DESCRIPTION COLUMNS USING THE NLTK LIBRARY

# ## DATA ANALYTICS
# 
# ### CATEGORY ANALYSIS
# 
# ### DESCRIPTIVE STATISTICS

# In[ ]:


df.describe()


# #### 1. ON AVERAGE, THE DISCOUNTED PRICE IS APPROXIMATELY 47% LOWER THAN THE ACTUAL PRICE.
# 
# #### 2. THE STANDARD DEVIATION OF THE DISCOUNTED PRICE IS QUITE HIGH COMPARED TO MEAN, INDICATING A WIDE RANGE OF PRICE DISCOUNTS.
# 
# #### 3. THE MINIMUM VALUE FOR BOTH ACTUAL PRICE AND DISCOUNTED PRICE IS 39, WHICH IS THE LOWEST-PRICED PRODUCT IN THE DATASET.
# 
# #### 4. THE MAXIMUM VALUE FOR THE ACTUAL PRICE IS 139,900, WHICH IS SIGNIFICANTLY HIGHER THAN 75TH PERCENTILE VALUE, INDICATING THAT THERE ARE SOME HIGH PRICED PRODUCTS IN THE DATASET.
# 
# #### 5. THE MAXIMUM DISCOUNT PERCENTAGE IS 94%, INDICATING THAT SOME PRODUCTS ARE HEAVILY DISCOUNTED.
# 
# #### 6. THE MEAN RATING IS 4.1 OUT OF 5, WHICH IS GOOD.
# 
# #### 7. THE MAXIMUM NUMBER OF RATING COUNT IS 426973, indicating tha some products have a high number of ratings.
# 
# #### GROUPED THE DATA BY CATEGORY AND CALCULATED THE NUMBER OF DISCOUNTS, AVERAGE DISCOUNT PERCENTAGE, AND MEDIAN DISCOUNT PERCENTAGE FOR EACH CATEGORY

# In[ ]:


discount_stats = df.groupby('category').agg({'discount_percentage' : ['count','mean','median']})
discount_stats.columns = ['num_discounts','avg_discount_percentage','median_discount_percentage']


# In[ ]:


discount_stats


# ## VISUALIZATION

# In[ ]:


plt.figure(figsize = (8,6))
sns.histplot(df['discount_percentage'],bins = 20,kde = True)
plt.title("DISCOUNT PERCENTAGE'S HISTOGRAM")
plt.xlabel('DISCOUNT PERCENTAGE')
plt.ylabel('COUNT')
plt.show()


# In[ ]:


plt.figure(figsize = (8,6))
sns.scatterplot(x='actual_price',y='discounted_price',data=df)
plt.title('ACTUAL PRICE VS DISCOUNTED PRICE')
plt.xlabel('ACTUAL PRICE')
plt.ylabel('DISCOUNTED PRICE')
plt.show()


# In[ ]:


plt.figure(figsize=(12, 8))
top10_categories = df['category_code'].value_counts().head(10).index
top10_categories_discount = df.groupby('category_code')['discount_percentage'].mean().loc[top10_categories]
sns.barplot(x=top10_categories, y=top10_categories_discount)
plt.xticks(rotation=90)
plt.title('Average Discount Percentage by Top 10 Categories')
plt.xlabel('Category')
plt.ylabel('Average Discount Percentage')
plt.show()


# In[ ]:


numeric_columns = df.select_dtypes(include=['int16', 'int32', 'int64', 'float64']).columns
corr = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:


plt.figure(figsize = (10,8))
sns.scatterplot(x='after_discount',y='rating',data=df)
plt.title('RELATION BETWEEN PRICE AND RATINGD')
plt.xlabel('PRICE AFTER DISCOUNT')
plt.ylabel('RATINGS')
plt.show()


# In[ ]:


ax = df['rating'].value_counts().sort_index().plot(
    kind = 'bar',
    title = 'COUNT OF RATINGS',
    figsize = (10,5)
)
ax.set_ylabel('COUNT')
ax.set_xlabel('RATINGS')
plt.show()


# ### THIS CHART INDICATES THAT MOST RATINGS ARE BETWEEN 3.8 AND 4.4, WITH A PEAK AT 4.1, SUGGESTING THAT MOST CUSTOMERS ARE SATISFIED WITH THE PRODUCTS.
# 
# ### THERE ARE VERY FEW RATINGS BELOW 3.8, INDICATING THAT CUSTOMERS ARE GENERALLY HAPPY WITH THE PRODUCTS THEY PURCHASE.

# # DATA SCIENCE

# In[ ]:


df['index'] = range(1,len(df)+1)
df.insert(0,'index',df.pop('index'))


# In[ ]:


df.head()


# In[ ]:


example = df['review_title'][62]
print(example)


# In[ ]:


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')


# In[ ]:


tokens = nltk.word_tokenize(example)


# In[ ]:


tokens[:10]


# ### USED NLTK.POS_TAG(TOKENS) FUNCTION TO TAG EACH TOKEN IN A GIVEN LIST OF TOKENS WITH ITS POS CATEGORY, SUCH AS NOUN, VERB, ADJECTIVE, ADVERB ETC.

# In[ ]:


tagged = nltk.pos_tag(tokens)


# In[ ]:


tagged[:10]


# In[ ]:


entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# In[41]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm


# In[42]:


sla = SentimentIntensityAnalyzer()


# In[43]:


sla.polarity_scores(example)


# In[44]:


res = {}

for i,row in tqdm(df.iterrows(),total = len(df)):
    text = row['review_title']
    myid = row['index']
    res[myid] = sla.polarity_scores(text)


# In[45]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns = {'index' : 'index'})
vaders = vaders.merge(df,how = 'left')


# In[46]:


vaders


# In[47]:


ax = sns.barplot(data=vaders,x='rating',y='compound')
ax.set_title('COMPOUND SCORE BY AMAZON STAR REVIEW')
plt.show()


# In[48]:


fig, axs = plt.subplots(1,3,figsize = (22,4))
sns.barplot(data = vaders,x = 'rating',y = 'pos',ax = axs[0])
sns.barplot(data = vaders,x = 'rating',y = 'neu',ax = axs[1])
sns.barplot(data = vaders,x = 'rating',y = 'neg',ax = axs[2])
axs[0].set_title('Positive',fontsize = 16)
axs[1].set_title('Neutral',fontsize = 16)
axs[2].set_title('Negative',fontsize = 16)

for ax in axs:
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=14)

plt.tight_layout()
plt.show()


# In[49]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


# ### 1. USED PRE-TRAINED MODEL FROM THE HUGGING FACE TRANSFORMERS LIBRARY
# 
# ### 2. LOADED THE TOKENIZER FOR THE PRE-TRAINED MODEL SPECIFIED IN MODEL.

# In[50]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[51]:


print(example)
sla.polarity_scores(example)


# In[52]:


encoded_text = tokenizer(example,return_tensors = 'pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)


# In[53]:


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example,return_tensors = 'pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
    }
    
    return scores_dict


# In[54]:


res = {}

for i,row in tqdm(df.iterrows(),total = len(df)):
    try:
        text = row['review_title']
        myid = row['index']
        vader_result = sla.polarity_scores(text)
        vader_result_rename = {}
        for key,value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f"Broke for id {myid}")


# In[55]:


results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index':'index'})
results_df = results_df.merge(df,how='left')


# In[56]:


results_df.columns


# In[57]:


sns.pairplot(data = results_df,
             vars = ['vader_neg','vader_neu','vader_pos','roberta_neg','roberta_neu','roberta_pos'],
             hue = 'rating',
             palette = 'tab10'
)
plt.show()


# In[58]:


results_df.query('rating==2').sort_values('roberta_pos',ascending=False)['review_title'].values[0]


# In[59]:


results_df.query('rating==3').sort_values('vader_pos',ascending=False)['review_title'].values[0]


# #### 1.THE CODE ABOVE QUERIES THE RESULTS_DF DATAFRAME TO RETRIEVE REVIEW TITLES WITH THE HIGHEST POSITIVE SENTIMENT SCORE FOR A SPECIFIC RATING.
# 
# #### 2.THE FIRST LINE RETRIEVES THE REVIEW TITLE WITH THE HIGHEST POSITIVE SENTIMENT SCORE FOR  A RATING OF 2.
# 
# #### 3.IT SORTS THE DATAFRAME BY THE ROBERTA_POS COLUMN IN DESCENDING ORDER TO GET THE TOP ROW OF THE DATAFRAME.
# 
# #### 4.THE SECOND LINE RETRIEVES THE REVIEW_TITLE WITH THE HIGHEST POSITIVE SENTIMENT SCORE FOR A RATING OF 3.
# 
# #### 5.IT SORTS THE DATAFRAME BY VADER_POS COLUMN IN DESCENDING ORDER TO GET THE TOP ROW OF THE DATAFRAME.

# In[61]:


from transformers import pipeline

sent_pipeline = pipeline('sentiment-analysis')


# In[ ]:


sent_pipeline(example)


# # SUMMARY OF THE STEPS I HAVE TAKEN FOR THE SENTIMENT ANALYSIS.
# 
# #### 1.IMPORTED NECESSARY PACKAGES SUCH AS PANDAS, SEABORN, NLTK, TRANSFORMERS ETC.
# 
# #### 2.READ IN A DATASET CONTAINING CUSTOMER REVIEWS OF VARIOUS PRODUCTS USING PANDAS.
# 
# #### 3.PERFORMED EDA ON DATASET TO GAIN INSIGHTS AND VISUALIZED THE DISTRIBUTION OF RATINGS USING SEABORN.
# 
# #### 4.USED THE NLTK LIBRARY TO PERFORM NLP TAKS SUCH AS TOKENIZATION, PART-OF-SPEECH TAGGING, AND NAMED ENTITY RECOGNITION ON THE SENTIMENT SCORES FOR EACH REVIEW TEXT.
# 
# #### 5.USED THE VADER SENTIMENT ANALYZER FOR NLTK.SENTIMENT LIBRARY TO COMPUTER SENTIMENT SCORES FOR EACH REVIEW TEXT.
# 
# #### 6.USED THE TRANSFORMERS LIBRARY TO LOAD AND USE A PRE-TRAINED SENTIMENT ANALYSIS MODEL FOR PREDICTING THE SENTIMENT OF THE REVIEW TEXT.
# 
# #### 7.VISUALIZED THE SENTIMENT SCORES AND THE DISTRIBUTION OF POSITIVE,NEGATIVE AND NEUTRAL SENTIMENT FOR EACH RATING USING SEABORN.
# 
# #### 8.RETRIEVED THE REVIEW TITLE WITH THE HIGHEST POSITIVE SENTIMENT SCORE FOR A SPECIFIC RATING BY QUERYING THE RESULTS DATAFRAME.
# 
# ### FINALLY, USED THE TRANSFORMERS LIBRARY AGAIN TO CREATE A SENTIMENT ANALYSIS PIPELINE FOR PREDICTING THE SENTIMENT OF NEW TEXT

# In[ ]:


reviews = list(df['review_title'])


# In[ ]:


for i in reviews:
    print(sent_pipeline(i))


# In[ ]:


sent_pipeline(reviews)


# ##### 1. average rating was about 4.1 out of 5, indicating a positive sentiment.
# ##### 2. the top 3 most reviewed categories were electronics, wearables and home appliances.
# ##### 3. sentiment analysis shows that mostly reviews were positive

# In[ ]:




