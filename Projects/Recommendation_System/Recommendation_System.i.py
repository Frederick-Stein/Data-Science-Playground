import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


## get data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']

url1 = 'https://raw.githubusercontent.com/Frederick-Stein/Data-Science-Playground/refs/heads/main/Projects/Recommendation_System/file.tsv'
utl2 = 'https://raw.githubusercontent.com/Frederick-Stein/Data-Science-Playground/refs/heads/main/Projects/Recommendation_System/Movie_Id_Titles.csv'
df = pd.read_csv(url1, sep='\t', names=column_names)
movie = pd.read_csv(utl2)
data = pd.merge(df, movie, on='item_id')
data.head()


## EDA
# data = data.drop(['timestamp', 'item_id'], axis=1)
print(data.shape)
data.info()


# prepare data
ratings = data.groupby('title').agg(
    rating = ('rating', lambda x: x.mean().round(2)),
    num = ('rating',  'count')
).reset_index()
print(ratings.sort_values('rating', ascending =False).head())
print(ratings.sort_values('num', ascending =False).head())

# visulaization
sns.set_style('whitegrid')
plt.figure(figsize = (10,4))
ratings['num'].hist(bins = 60)
plt.title('Number of Ratings')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Movies')
plt.show()
plt.title('Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Number of Movies')
ratings['rating'].hist(bins = 60)
plt.show()


# matrix of movie
moviemat = data.pivot_table(index='user_id', columns='title', values='rating')
moviemat.head()


# compute corrlation
def corr_with(R, target_col, min_overlap=5):
    t = R[target_col]
    mask = t.notna()     # users who rated target
    Rsub = R.loc[mask]
    tsub = t[mask]

    # keep items with enough co-raters in this overlap
    valid = Rsub.notna().sum(axis=0) >= min_overlap
    Rsub = Rsub.loc[:, valid]

    # drop items whose ratings are constant in the overlap (std = 0)
    nzstd = Rsub.std(skipna=True) > 0
    Rsub = Rsub.loc[:, nzstd]

    # compute Pearson correlations
    corr = Rsub.corrwith(tsub).dropna().to_frame('Correlation')

    return corr.sort_values('Correlation', ascending=False)

corr_starwars = corr_with(moviemat, "Star Wars (1977)", min_overlap=20)
corr_starwars = corr_starwars.merge(ratings, on='title', how = 'left')
corr_starwars[corr_starwars['num'] > 100].head(10)
