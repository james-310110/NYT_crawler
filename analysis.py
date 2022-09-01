import csv
import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from textblob import TextBlob
from happytransformer import HappyTextClassification
from scipy.signal import savgol_filter as smooth
from sklearn.preprocessing import MinMaxScaler 
from scipy.signal import find_peaks
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def combine_mega(start,end,path,df_mega,csv_file):
    for i in range(start,end+1):
        fpath = path+'mega'+str(i/1)+'.csv'
        df = pd.read_csv(fpath)
        df_tot = pd.concat([df_mega,df]).drop_duplicates().reset_index(drop=True)
    df_tot.to_csv(csv_file,index=False)
    return df_tot

def measure_relevance(df_mega, fpath):
    df_mega['relevance'] = None
    target_words = ['China','Chinese','Beijing','Shanghai','Hong Kong','Wuhan','Xi','Mao','Communist']
    for idx in df_mega.index:
        headline = df_mega['headline'][idx]
        description = df_mega['description'][idx]
        tags = df_mega['tags'][idx]
        score = 0
        for word in target_words:
            if not pd.isnull(headline) and word in headline:
                score += 2
            if not pd.isnull(description) and word in description:
                score += 1
            if word in tags:
                score += 1
        df_mega['relevance'][idx] = score
    irrelevant_indices = df_mega[df_mega['relevance']<2].index
    df_mega.drop(irrelevant_indices, inplace=True)
    df_mega.to_pickle(fpath)
    return df_mega

def clean_text(text, stop_words, lemmatizer):
    text = text.lower()
    text = ' '.join([word for word in text.split(' ') if word not in stop_words])
    text = text.encode('ascii', 'ignore').decode()
    text = re.sub(r'https*\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'\'\w+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\w*\d+\w*', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = [lemmatizer.lemmatize(w,'v') for w in nltk.word_tokenize(text)]
    return ' '.join(text)

def tokenize_text(df_mega,fpath):
    stop_words = stopwords.words('english')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    df_mega['headline_tokens'] = ''
    df_mega['description_tokens'] = ''
    df_mega['tags_tokens'] = ''
    df_mega['content_tokens'] = ''
    for idx in df_mega.index:
        headline = df_mega['headline'][idx]
        description = df_mega['description'][idx]
        tags = df_mega['tags'][idx]
        content = df_mega['tags'][idx]
        if not pd.isnull(headline):
            df_mega['headline_tokens'][idx] = clean_text(headline, stop_words, lemmatizer)
        if not pd.isnull(description):
            df_mega['description_tokens'][idx] = clean_text(description, stop_words, lemmatizer)
        if not pd.isnull(tags):
            df_mega['tags_tokens'][idx] = clean_text(tags, stop_words, lemmatizer)
        if not pd.isnull(content):
            df_mega['content_tokens'][idx] = clean_text(content, stop_words, lemmatizer)
    df_mega.to_pickle(fpath)
    return df_mega

def textblob_analyze(df_mega,fpath):
    df_mega['tb_polarity'] = None # [-1.0,1.0]
    df_mega['tb_subjectivity'] = None # [0.0,1.0]
    for idx in df_mega.index:
        text = df_mega['content'][idx]
        df_mega['tb_polarity'][idx] = TextBlob(text).sentiment.polarity
        df_mega['tb_subjectivity'][idx] = TextBlob(text).sentiment.subjectivity
    
    df_mega.to_pickle(fpath)
    return df_mega

def happytransformer_analyze(df_mega,fpath):
    df_mega['hp_polarity'] = None # [-1.0,1.0]
    df_mega['hp_intensity'] = None # [0.0,1.0]
    happy_tc = HappyTextClassification(model_type="DISTILBERT",  model_name="distilbert-base-uncased-finetuned-sst-2-english")
    for idx in df_mega.index:
        text = df_mega['content'][idx]
        text = text.encode('ascii', 'ignore').decode()
        text = re.sub(r'https*\S+', ' ', text)
        text = re.sub(r'@\S+', ' ', text)
        text = re.sub(r'#\S+', ' ', text)
        text = re.sub(r'\'\w+', '', text)
        text = re.sub(r'\w*\d+\w*', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        sentences = nltk.sent_tokenize(text)
        hp_polarity, hp_intensity = 0, 0
        for sent in sentences:
            hp_result = happy_tc.classify_text(sent[:512])
            hp_intensity += hp_result.score
            if hp_result.label == 'NEGATIVE':
                hp_polarity -= 1
            elif hp_result.label == 'POSITIVE':
                hp_polarity += 1
        if len(sentences) == 0:
            df_mega['hp_polarity'][idx] = None
            df_mega['hp_intensity'][idx] = None
        else:
            df_mega['hp_polarity'][idx] = hp_polarity / len(sentences)
            df_mega['hp_intensity'][idx] = hp_intensity / len(sentences)
    df_mega.to_pickle(fpath)
    return df_mega
    
def organize_by_time(df_mega,fpath):
    df_time_frequency = df_mega.groupby(['date'],as_index=False)['relevance'].count()
    df_time_sentiment = df_mega.groupby(['date'],as_index=False)[
        'tb_polarity','tb_subjectivity'].mean()
    df_time_tokens = df_mega.groupby(['date'],as_index=False) \
        .agg({'headline_tokens': ' '.join,
              'description_tokens': ' '.join,
              'tags_tokens': ' '.join,
              'content_tokens': ' '.join})
    df_time = pd.merge(df_time_sentiment,df_time_tokens, on='date')
    df_time = pd.merge(df_time, df_time_frequency, on='date').rename(columns={'relevance':'frequency'})
    # df_time.rename({'relevance':'frequency'},inplace=True)
    # df_time['frequency'] = df_mega.groupby(['date'],as_index=False).count()
    # df_time = pd.merge(df_time,df_mega.groupby(['date'],as_index=False).count(), on='date')
    df_time.to_pickle(fpath)
    return df_time

def graph_timeseries_sentiment(df_time):
    x = df_time['date']
    
    # frequency_ra = df_time['frequency'].rolling(91).sum()
    # tb_polarity_ra = df_time['tb_polarity'].rolling(91).sum()
    # tb_subjectivity_ra = df_time['tb_subjectivity'].rolling(91).sum()
    
    scaler = MinMaxScaler()
    frequency_ra = np.ravel(scaler.fit_transform(df_time['frequency'].rolling(91).mean().values.reshape(-1,1)))
    tb_polarity_ra = np.ravel(scaler.fit_transform(df_time['tb_polarity'].rolling(91).mean().values.reshape(-1,1)))
    tb_subjectivity_ra = np.ravel(scaler.fit_transform(df_time['tb_subjectivity'].rolling(91).mean().values.reshape(-1,1)))
    frequency_ra = smooth(frequency_ra,50,3)
    tb_polarity_ra = smooth(tb_polarity_ra,50,3)
    tb_subjectivity_ra = smooth(tb_subjectivity_ra,50,3)
    
    frequency_peaks, _ = find_peaks(frequency_ra,prominence=0.1,distance=14)
    frequency_mins, _ = find_peaks(frequency_ra*-1,prominence=0.1,distance=14)
    tb_polarity_peaks, _ = find_peaks(tb_polarity_ra,prominence=0.1, distance=14)
    tb_polarity_mins, _ = find_peaks(tb_polarity_ra*-1,prominence=0.1, distance=14)
    tb_subjectivity_peaks, _ = find_peaks(tb_subjectivity_ra,prominence=0.1, distance=14)
    tb_subjectivity_mins, _ = find_peaks(tb_subjectivity_ra*-1,prominence=0.1, distance=14)
    
    fig, ax = plt.subplots()
    
    # plt.plot(x,frequency_ra,color='black',label='frequency scores')
    # plt.plot(x[frequency_mins],frequency_ra[frequency_mins], 'x', markersize=10, label='frequency mins')
    # plt.plot(x[frequency_peaks], frequency_ra[frequency_peaks], '*', markersize=10, label='frequency peaks')
    
    # plt.plot(x, tb_polarity_ra,color='red',label='polarity scores')
    # plt.plot(x[tb_polarity_mins],tb_polarity_ra[tb_polarity_mins], 'x', markersize=10, label='polarity mins')
    # plt.plot(x[tb_polarity_peaks], tb_polarity_ra[tb_polarity_peaks], '*', markersize=10, label='polarity peaks')
    
    plt.plot(x, tb_subjectivity_ra,color='blue',label='subjectivity scores')
    plt.plot(x[tb_subjectivity_mins],tb_subjectivity_ra[tb_subjectivity_mins], 'x', markersize=10, label='subjectivity mins')
    plt.plot(x[tb_subjectivity_peaks], tb_subjectivity_ra[tb_subjectivity_peaks], '*', markersize=10, label='subjectivity peaks')
    
    leg = ax.legend()
    plt.xticks(x[::91],rotation='vertical')
    plt.xlabel('Date')
    plt.ylabel('Sentiment')
    # plt.savefig('result/timeseries.png')
    plt.show()
    # return pd.concat([x[tb_polarity_mins],x[tb_subjectivity_mins]],axis=0).drop_duplicates(), \
    #        pd.concat([x[tb_polarity_peaks],x[tb_subjectivity_peaks]],axis=0).drop_duplicates()
    return x[tb_polarity_mins],x[tb_polarity_peaks]
    
def generate_wordcloud(df_time, neg_dates, pos_dates):
    neg_tokens = df_time.loc[df_time['date'].isin(neg_dates)]
    pos_tokens = df_time.loc[df_time['date'].isin(pos_dates)]
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 50,
        max_font_size = 64,
        scale = 3,
        random_state = 1
    )
    # for _, row in neg_tokens.iterrows():
    #     # print(row.shape)
    #     wordcloud.generate(row['tags_tokens'])
    #     wordcloud.to_file('result/neg_tags_wordcloud'+str(row['date'])+'.png')
    # for _, row in pos_tokens.iterrows():
    #     wordcloud.generate(row['tags_tokens'])
    #     wordcloud.to_file('result/pos_tags_wordcloud'+str(row['date'])+'.png')
    for _, row in neg_tokens.iterrows():
        # print(row.shape)
        wordcloud.generate(row['content_tokens'])
        wordcloud.to_file('result/neg_content_wordcloud'+str(row['date'])+'.png')
    for _, row in pos_tokens.iterrows():
        wordcloud.generate(row['content_tokens'])
        wordcloud.to_file('result/pos_content_wordcloud'+str(row['date'])+'.png')
    for _, row in neg_tokens.iterrows():
        # print(row.shape)
        wordcloud.generate(row['headline_tokens'])
        wordcloud.to_file('result/neg_headline_wordcloud'+str(row['date'])+'.png')
    for _, row in pos_tokens.iterrows():
        wordcloud.generate(row['headline_tokens'])
        wordcloud.to_file('result/pos_headline_wordcloud'+str(row['date'])+'.png')


############ hyperparameters ############
csv_file = 'data/mega.csv'
mega_pickle = 'data/df_mega.pkl'
timeseries_pickle = 'data/df_time.pkl'
section_pickle = 'data/df_section.pkl'

############ loading temp files #############
# df_mega = pd.read_csv(csv_file)
df_mega = pd.read_pickle(mega_pickle)
df_time = pd.read_pickle(timeseries_pickle)

############ action codes ############
# df_mega = combine_mega(1,45,'data/',df_mega,csv_file)
# print('mega data combined')
# df_mega = measure_relevance(df_mega,mega_pickle)
# print('relevance of articles measured')
# df_mega = textblob_analyze(df_mega, mega_pickle)
# print('sentiment analysis completed by textblob')
# df_mega = tokenize_text(df_mega, mega_pickle)
# print('text tokenized by nltk')
# df_mega = happytransformer_analyze(df_mega,mega_pickle)
# print('sentiment analysis completed by happytransformer')
# df_time = organize_by_time(df_mega,timeseries_pickle)
# print('timeseries data compiled')
neg_dates, pos_dates = graph_timeseries_sentiment(df_time)
print('critical dates found')
# generate_wordcloud(df_time,neg_dates,pos_dates)
# print('wordcloud generated for critical dates')

print(df_mega.shape,df_mega.columns)
print(df_time.shape,df_time.columns)