import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

geolocator = Nominatim()

# Import files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
combine = [train, test]

combine[0] = combine[0].drop(['id'], axis = 1)


# Filling in missing cities
# count = 0
# for df in combine:
#     df.loc[(df['city'].isnull()), 'city'] = "temp"
#
#     for index, row in df.iterrows():
#         if row['city'] == "temp":
#             city = geolocator.reverse(str(row['latitude']) + "," + str(row['longitude']), timeout=10).raw['address']['city']
#             df.loc[index, 'city'] = city
#
#     if (count == 0):
#         df.to_pickle('city_train.pickle')
#     else:
#         df.to_pickle('city_test.pickle')
#
#     count += 1;


combine[0]['city'] = pd.read_pickle('city_train.pickle')
combine[1]['city'] = pd.read_pickle('city_test.pickle')

combine[0]['zipcode'] = pd.read_pickle('zip_train3.pickle')
combine[1]['zipcode'] = pd.read_pickle('zip_test3.pickle')


# count = 0
# for df in combine:
#     df["zipcode"] = df["zipcode"].str[:5]
#
#     for index, row in df.iterrows():
#         if (row['zipcode'] == "temp") | (not row['zipcode'].isdigit()):
#             zip = geolocator.reverse(str(row['latitude']) + "," + str(row['longitude']), timeout=10).raw['address']['postcode']
#             df.loc[index, 'zipcode'] = zip
#
#     # Fixing long zip codes and replacing still unknown ones with new number
#     # df["zipcode"] = df["zipcode"].str[:5]
#     # df.loc[(df['zipcode'] == "temp"), 'zipcode'] = -1
#     #df['zipcode'] = df['zipcode'].astype(int)
#     # df.loc[(df['zipcode'] == -1), 'zipcode'] = df['zipcode'].max()+1
#
#     if (count == 0):
#         df['zipcode'].to_pickle('zip_train2.pickle')
#     else:
#         df['zipcode'].to_pickle('zip_test2.pickle')
#
#     count += 1;

# count = 0
# for df in combine:
#
#     for index, row in df.iterrows():
#         if (not row['zipcode'].isdigit()):
#             if (row['zipcode'][:1] == 'M'):
#                 df.loc[index, 'zipcode'] = row['zipcode'][-5:]
#             elif (row['zipcode'][:1] == 'C'):
#                 df.loc[index, 'zipcode'] = row['zipcode'][-5:]
#             else:
#                 df.loc[index, 'zipcode'] = row['zipcode'][:5]
#
#
#
# count = 0
# for df in combine:
#     df['zipcode'] = df['zipcode'].astype(int)
#     if (count == 0):
#         df['zipcode'].to_pickle('zip_train3.pickle')
#     else:
#         df['zipcode'].to_pickle('zip_test3.pickle')
#
#     count += 1;


# print(train['accommodates'].unique())

property_type_mapping = {"Apartment": 1, "House": 2, "Condominium": 3, "Loft": 4, "Hostel": 5, "Guest suite": 6,
                         "Bed & Breakfast": 7, "Bungalow": 8, "Guesthouse": 9, "Dorm": 10, "Other": 11,
                         "Camper/RV": 12, "Villa": 13, "Boutique hotel": 14, "Timeshare": 15, "In-law":16, "Boat":17,
                         "Serviced apartment":18, "Castle":19, "Cabin":20, "Treehouse":21, "Tipi":22, "Vacation home":23,
                         "Tent":24, "Hut":25, "Casa particular":26, "Chalet":27, "Yurt":28, "Earth House":29,
                         "Parking Space":30, "Train":31, "Cave":32, "Lighthouse":33, "Island":34, "Townhouse":35, np.nan: 36
                         }

room_type_mapping = {"Entire home/apt": 1, "Private room": 2, "Shared room": 3, np.nan: 4}

bed_type_mapping = {"Real Bed":1, "Futon":2, "Pull-out Sofa":3, "Couch":4, "Airbed":5, np.nan: 6}

cancel_policy_mapping = {"strict":1, "moderate":2, "flexible":3, "super_strict_30":4, "super_strict_60":5, np.nan: 6}

cleaning_mapping = {"TRUE":1, "FALSE": 0, np.nan: 2}

city_mapping = {"NYC":1, "SF": 2, "DC": 3, "Washington": 3, "LA": 4, "Chicago": 5, "Boston": 6, np.nan: 7}

profile_pic_mapping = {"t": 1, "f": 2, np.nan: 3}

identity_mapping = {"t": 1, "f": 2, np.nan: 3}

bookable_mapping = {"t": 1, "f": 2, np.nan: 3}




# Apply some manual mappings
for df in combine:
    df["property_type"] = df["property_type"].map(property_type_mapping)
    df["room_type"] = df["room_type"].map(room_type_mapping)
    df["bed_type"] = df["bed_type"].map(bed_type_mapping)
    df["cancellation_policy"] = df["cancellation_policy"].map(cancel_policy_mapping)
    df["cleaning_fee"] = df["cleaning_fee"].map(cleaning_mapping)
    df["city"] = df["city"].map(city_mapping)
    df["host_has_profile_pic"] = df["host_has_profile_pic"].map(profile_pic_mapping)
    df["host_identity_verified"] = df["host_identity_verified"].map(identity_mapping)
    df["instant_bookable"] = df["instant_bookable"].map(bookable_mapping)


# Fill in some missing with best guesses

for df in combine:
    df.loc[(df['cleaning_fee'].isnull()), 'cleaning_fee'] = 2
    df.loc[(df['cancellation_policy'].isnull()), 'cancellation_policy'] = 6

for df in combine:
    df['host_response_rate'] = df['host_response_rate'].str.rstrip('%').astype('float') / 100.0


score = np.zeros(4)

for i in range(1, 4):
    score_df = combine[0][(combine[0]['instant_bookable'] == i)]['host_response_rate'].dropna()
    score_med = score_df.mean()

    if np.isnan(score_med):
        score[i] = 0
    else:
        score[i] = score_med


for df in combine:
    for i in range(1, 4):
        df.loc[(df['host_response_rate'].isnull()) & (df['instant_bookable'] == i), 'host_response_rate'] = score[i]

    df['host_response_rate'] = df['host_response_rate'].astype(np.float64)


combine[1].loc[(combine[1]['accommodates'].isnull()), 'accommodates'] = 0

guess_bathrooms = np.zeros(int(combine[0]['accommodates'].max()+1))
guess_bedrooms = np.zeros(int(combine[0]['accommodates'].max()+1))
guess_beds = np.zeros(int(combine[0]['accommodates'].max()+1))

for i in range(0, combine[0]['accommodates'].max()+1):
    bathroom_df = combine[0][(combine[0]['accommodates'] == i)]['bathrooms'].dropna()
    bedroom_df = combine[0][(combine[0]['accommodates'] == i)]['bedrooms'].dropna()
    beds_df = combine[0][(combine[0]['accommodates'] == i)]['beds'].dropna()


    bathroom_med = bathroom_df.median()
    bedroom_med = bedroom_df.median()
    beds_med = beds_df.median()

    if np.isnan(bathroom_med):
        guess_bathrooms[i] = 0
    else:
        guess_bathrooms[i] = bathroom_med

    if np.isnan(bedroom_med):
        guess_bedrooms[i] = 0
    else:
        guess_bedrooms[i] = bedroom_med

    if np.isnan(beds_med):
        guess_beds[i] = 0
    else:
        guess_beds[i] = beds_med


for ds in combine:
    for i in range(0, ds['accommodates'].max()+1):
        ds.loc[(ds['bathrooms'].isnull()) & (ds['accommodates'] == i), 'bathrooms'] = guess_bathrooms[i]
        ds.loc[(ds['bedrooms'].isnull()) & (ds['accommodates'] == i), 'bedrooms'] = guess_bedrooms[i]
        ds.loc[(ds['beds'].isnull()) & (ds['accommodates'] == i), 'beds'] = guess_beds[i]

    ds['bathrooms'] = ds['bathrooms'].astype(int)
    ds['bedrooms'] = ds['bedrooms'].astype(int)
    ds['beds'] = ds['beds'].astype(int)



combine[0] = combine[0].drop(['review_scores_rating'], axis=1)
combine[1] = combine[1].drop(['review_scores_rating'], axis=1)

# Number of days open
for ds in combine:
    ds.loc[(ds['host_since'].isnull()), 'host_since'] = dt.datetime(2018,2,12)
    ds['host_since'] = ds['host_since'].astype('datetime64[ns]')
    ds['days_host'] = (dt.datetime(2018, 2, 12) - ds['host_since']).dt.days
    ds['days_host'] = ds['days_host'].astype(int)
    ds['days_host'] = np.log1p(ds['days_host'])


# Average reviews per day starting at first review
for df in combine:
    df.loc[(df['first_review'].isnull()), 'first_review'] = dt.datetime(2018, 2, 12)
    df.loc[(df['last_review'].isnull()), 'last_review'] = dt.datetime(2018, 2, 12)
    df['first_review'] = df['first_review'].astype('datetime64[ns]')
    df['last_review'] = df['last_review'].astype('datetime64[ns]')
    df.loc[(df['number_of_reviews'].isnull()), 'number_of_reviews'] = 0
    # df['number_of_reviews'].fillna(0, inplace=True)
    df['temp'] = (df['last_review'] - df['first_review']).dt.days
    df['temp'] = df['temp'].astype(int)
    df.loc[(df['temp'] == 0), 'temp'] = 1
    df['review_rate'] = df['number_of_reviews'] / df['temp']
    df = df.drop(['temp'], axis = 1)

    df.loc[(df['review_rate'] == np.NaN), 'review_rate'] = 0


# Getting missing zipcodes from lat/lon
# count = 0
# for df in combine:
#     df.loc[(df['zipcode'].isnull()) | (len(df['zipcode']) < 5), 'zipcode'] = "temp"
#     # df['zipcode'] = df.apply(
#     #     lambda row: (geolocator.reverse(str(row['latitude'])+","+str(row['longitude']),timeout=10).raw['address']['postcode'])
#     #     if row['zipcode'] == "temp" else row['zipcode'],
#     #     axis=1
#     # )
#
#     for index, row in df.iterrows():
#         if row['zipcode'] == "temp":
#             zip = geolocator.reverse(str(row['latitude']) + "," + str(row['longitude']), timeout=10).raw['address']['postcode']
#             df.loc[index, 'zipcode'] = zip
#
#     # Fixing long zip codes and replacing still unknown ones with new number
#     # df["zipcode"] = df["zipcode"].str[:5]
#     # df.loc[(df['zipcode'] == "temp"), 'zipcode'] = -1
#     #df['zipcode'] = df['zipcode'].astype(int)
#     # df.loc[(df['zipcode'] == -1), 'zipcode'] = df['zipcode'].max()+1
#
#     if (count == 0):
#         df['zipcode'].to_pickle('zip_train.pickle')
#     else:
#         df['zipcode'].to_pickle('zip_test.pickle')
#
#     count += 1;


#df = pd.read_pickle('zip')


# Transforming Lat/Lon
coords = np.vstack((combine[0][['latitude', 'longitude']].values,
                   combine[1][['latitude', 'longitude']].values))

pca = PCA().fit(coords)

for df in combine:
    df['pca0'] = pca.transform(df[['latitude', 'longitude']])[:, 0]
    df['pca1'] = pca.transform(df[['latitude', 'longitude']])[:, 1]



# Clustering
num_clusters = 500
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=10000, random_state=34645).fit(coords[sample_ind])

for df in combine:
    df.loc[:, 'loc_cluster'] = kmeans.predict(df[['latitude', 'longitude']])



# Plotting clusters
# city_long_border = (combine[0]['longitude'].max()+0.25,combine[0]['longitude'].min()-0.25)
# city_lat_border = (combine[0]['latitude'].max()+0.25,combine[0]['latitude'].min()-0.25)
# N=74000
# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.scatter(combine[0].longitude.values[:N], combine[0].latitude.values[:N], s=10, lw=0,
#            c=combine[0].loc_cluster[:N].values, cmap='tab20', alpha=0.2)
# ax.set_xlim(city_long_border)
# ax.set_ylim(city_lat_border)
# ax.set_xlabel('Longitude (Use zoom tool on clusters)')
# ax.set_ylabel('Latitude')
# plt.show()



# Change amenities to word vectors
def review_to_words(s):
    if s is None:
        return ""
    if (type(s) != str):
        s = str(s)
    # 1. Remove HTML
    s_text = BeautifulSoup(s, "html.parser").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", s_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. Convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return(" ".join( meaningful_words ))


# Vectorizing amenities
count = 0
combine[0].loc[(combine[0]['amenities'].isnull()), 'amenities'] = ""
combine[1].loc[(combine[1]['amenities'].isnull()), 'amenities'] = ""

num_reviews = combine[0]["amenities"].size
clean_train_reviews = []
for i in range(0, num_reviews):
      # if ((i + 1) % 10000 == 0):
     #     print("Row %d of %d\n" % (i + 1, num_reviews))
    clean_train_reviews.append(review_to_words(combine[0]["amenities"][i]))

vectorizer = CountVectorizer(analyzer="word", \
                            tokenizer=None, preprocessor=None, stop_words='english', max_features=50)


train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray()

add = pd.DataFrame(train_data_features)

# if (count == 0):
#     add.to_pickle('amenities_train.pickle')
# else:
#     add.to_pickle('amenities_test.pickle')
#
combine[0] = pd.concat([combine[0], add], axis=1)


num_reviews = combine[1]["amenities"].size
clean_test_reviews = []
for i in range(0, num_reviews):
      # if ((i + 1) % 10000 == 0):
     #     print("Row %d of %d\n" % (i + 1, num_reviews))
    clean_test_reviews.append(review_to_words(combine[1]["amenities"][i]))

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
add = pd.DataFrame(test_data_features)
combine[1] = pd.concat([combine[1], add], axis=1)
#




# Vectorizing name
count = 0
combine[0].loc[(combine[0]['name'].isnull()), 'name'] = ""
combine[1].loc[(combine[1]['name'].isnull()), 'name'] = ""

num_reviews = combine[0]["name"].size
clean_train_reviews = []
for i in range(0, num_reviews):
      # if ((i + 1) % 10000 == 0):
     #     print("Row %d of %d\n" % (i + 1, num_reviews))
    clean_train_reviews.append(review_to_words(combine[0]["name"][i]))

vectorizer = CountVectorizer(analyzer="word", \
                            tokenizer=None, preprocessor=None, stop_words='english', max_features=100)


train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray()

add = pd.DataFrame(train_data_features)

# if (count == 0):
#     add.to_pickle('amenities_train.pickle')
# else:
#     add.to_pickle('amenities_test.pickle')
#
combine[0] = pd.concat([combine[0], add], axis=1)


num_reviews = combine[1]["name"].size
clean_test_reviews = []
for i in range(0, num_reviews):
      # if ((i + 1) % 10000 == 0):
     #     print("Row %d of %d\n" % (i + 1, num_reviews))
    clean_test_reviews.append(review_to_words(combine[1]["name"][i]))

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
add = pd.DataFrame(test_data_features)
combine[1] = pd.concat([combine[1], add], axis=1)



# Vectorizing Neighbourhood
# count = 0
# for df in combine:
#     df.loc[(df['neighbourhood'].isnull()), 'neighbourhood'] = ""
#     num_reviews = df["neighbourhood"].size
#     clean_train_reviews = []
#     for i in range(0, num_reviews):
#         # if ((i + 1) % 10000 == 0):
#         #     print("Row %d of %d\n" % (i + 1, num_reviews))
#         clean_train_reviews.append(review_to_words(df["neighbourhood"][i]))
#
#     vectorizer = CountVectorizer(analyzer="word", \
#                                  tokenizer=None, preprocessor=None, stop_words=None, max_features=250)
#
#
#     train_data_features = vectorizer.fit_transform(clean_train_reviews)
#
#     train_data_features = train_data_features.toarray()
#
#     add = pd.DataFrame(train_data_features)
#
#     if (count == 0):
#         add.to_pickle('neigh_train.pickle')
#     else:
#         add.to_pickle('neigh_test.pickle')
#
#     if (count == 0):
#         combine[0] = pd.concat([combine[0], add], axis=1)
#     else:
#         combine[1] = pd.concat([combine[1], add], axis=1)
#
#     count += 1;



# Vectorizing Description
count = 0
combine[0].loc[(combine[0]['description'].isnull()), 'description'] = ""
combine[1].loc[(combine[1]['description'].isnull()), 'description'] = ""

num_reviews = combine[0]["description"].size
clean_train_reviews = []
for i in range(0, num_reviews):
      # if ((i + 1) % 10000 == 0):
     #     print("Row %d of %d\n" % (i + 1, num_reviews))
    clean_train_reviews.append(review_to_words(combine[0]["description"][i]))

vectorizer = CountVectorizer(analyzer="word", \
                            tokenizer=None, preprocessor=None, stop_words='english', max_features=500)


train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray()

add = pd.DataFrame(train_data_features)

# if (count == 0):
#     add.to_pickle('amenities_train.pickle')
# else:
#     add.to_pickle('amenities_test.pickle')
#
combine[0] = pd.concat([combine[0], add], axis=1)


num_reviews = combine[1]["description"].size
clean_test_reviews = []
for i in range(0, num_reviews):
      # if ((i + 1) % 10000 == 0):
     #     print("Row %d of %d\n" % (i + 1, num_reviews))
    clean_test_reviews.append(review_to_words(combine[1]["description"][i]))

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
add = pd.DataFrame(test_data_features)
combine[1] = pd.concat([combine[1], add], axis=1)




# Create some simple additional features
for df in combine:
    df['totrooms'] = df['bedrooms'] + df['bathrooms']


combine[0]['cluster_only_price_med'] = 0
combine[1]['cluster_only_price_med'] = 0
guess_prices = np.zeros(int(combine[0]['loc_cluster'].max()+1))
for i in range(0, combine[0]['loc_cluster'].max()+1):
    medprice_df = combine[0][(combine[0]['loc_cluster'] == i)]['log_price'].dropna()

    price_med = medprice_df.median()

    if np.isnan(price_med):
        guess_prices[i] = 0
    else:
        guess_prices[i] = price_med

for df in combine:
    for i in range(0, num_clusters):
        df.loc[(df['loc_cluster'] == i), 'cluster_only_price_med'] = guess_prices[i]



combine[0]['cluster_only_price_mean'] = 0
combine[1]['cluster_only_price_mean'] = 0
guess_prices = np.zeros(int(combine[0]['loc_cluster'].max()+1))
for i in range(0, combine[0]['loc_cluster'].max()+1):
    medprice_df = combine[0][(combine[0]['loc_cluster'] == i)]['log_price'].dropna()

    price_med = medprice_df.mean()

    if np.isnan(price_med):
        guess_prices[i] = 0
    else:
        guess_prices[i] = price_med

for df in combine:
    for i in range(0, num_clusters):
        df.loc[(df['loc_cluster'] == i), 'cluster_only_price_mean'] = guess_prices[i]



combine[0]['cluster_price_med'] = 0
combine[1]['cluster_price_med'] = 0
guess_prices2 = np.zeros((int(num_clusters), int(combine[0]['accommodates'].max()+1)))

for i in range(0, num_clusters):
    for j in range(0, int(combine[0]['accommodates'].max()+1)):
        medprice_df = combine[0][(combine[0]['loc_cluster'] == i) & (combine[0]['accommodates'] == j)]['log_price'].dropna()

        price_med = medprice_df.median()

        if np.isnan(price_med):
            guess_prices2[i,j] = 0
        else:
            guess_prices2[i,j] = price_med

for df in combine:
    for i in range(0, num_clusters):
        for j in range(0, int(combine[0]['accommodates'].max() + 1)):
            df.loc[(df['loc_cluster'] == i) & (df['accommodates'] == j), 'cluster_price_med'] = guess_prices2[i,j]



combine[0]['cluster_price_mean'] = 0
combine[1]['cluster_price_mean'] = 0
guess_prices2 = np.zeros((int(num_clusters), int(combine[0]['accommodates'].max()+1)))

for i in range(0, num_clusters):
    for j in range(0, int(combine[0]['accommodates'].max()+1)):
        medprice_df = combine[0][(combine[0]['loc_cluster'] == i) & (combine[0]['accommodates'] == j)]['log_price'].dropna()

        price_med = medprice_df.mean()

        if np.isnan(price_med):
            guess_prices2[i,j] = 0
        else:
            guess_prices2[i,j] = price_med

for df in combine:
    for i in range(0, num_clusters):
        for j in range(0, int(combine[0]['accommodates'].max() + 1)):
            df.loc[(df['loc_cluster'] == i) & (df['accommodates'] == j), 'cluster_price_mean'] = guess_prices2[i,j]



combine[0]['cluster_price_med_city'] = 0
combine[1]['cluster_price_med_city'] = 0
guess_prices3 = np.zeros((int(num_clusters), int(combine[0]['beds'].max()+1)))

for i in range(0, num_clusters):
    for j in range(0, int(combine[0]['beds'].max()+1)):
        medprice_df = combine[0][(combine[0]['loc_cluster'] == i) & (combine[0]['beds'] == j)]['log_price'].dropna()

        price_med = medprice_df.median()

        if np.isnan(price_med):
            guess_prices3[i,j] = 0
        else:
            guess_prices3[i,j] = price_med

for df in combine:
    for i in range(0, num_clusters):
        for j in range(0, int(combine[0]['beds'].max() + 1)):
            df.loc[(df['loc_cluster'] == i) & (df['beds'] == j), 'cluster_price_med_city'] = guess_prices3[i,j]


combine[0]['cluster_price_mean_city'] = 0
combine[1]['cluster_price_mean_city'] = 0
guess_prices3 = np.zeros((int(num_clusters), int(combine[0]['beds'].max()+1)))

for i in range(0, num_clusters):
    for j in range(0, int(combine[0]['beds'].max()+1)):
        medprice_df = combine[0][(combine[0]['loc_cluster'] == i) & (combine[0]['beds'] == j)]['log_price'].dropna()

        price_med = medprice_df.mean()

        if np.isnan(price_med):
            guess_prices3[i,j] = 0
        else:
            guess_prices3[i,j] = price_med

for df in combine:
    for i in range(0, num_clusters):
        for j in range(0, int(combine[0]['beds'].max() + 1)):
            df.loc[(df['loc_cluster'] == i) & (df['beds'] == j), 'cluster_price_mean_city'] = guess_prices3[i,j]



# Rescale
for df in combine:
    df['zipcode'] = (df['zipcode']-df['zipcode'].min())/(df['zipcode'].max()-df['zipcode'].min())



# Dropping columns that are not needed

combine[0] = combine[0].drop(['amenities'], axis=1)
combine[0] = combine[0].drop(['name'], axis=1)
combine[0] = combine[0].drop(['neighbourhood'], axis=1)
combine[0] = combine[0].drop(['description'], axis=1)
combine[0] = combine[0].drop(['thumbnail_url'], axis=1)
combine[0] = combine[0].drop(['latitude'], axis=1)
combine[0] = combine[0].drop(['longitude'], axis=1)
combine[0] = combine[0].drop(['host_has_profile_pic'], axis=1)
combine[0] = combine[0].drop(['host_identity_verified'], axis=1)
combine[0] = combine[0].drop(['first_review'], axis=1)
combine[0] = combine[0].drop(['host_since'], axis=1)
combine[0] = combine[0].drop(['last_review'], axis=1)
combine[0] = combine[0].drop(['cleaning_fee'], axis=1)
combine[0] = combine[0].drop(['bathrooms'], axis=1)
combine[0] = combine[0].drop(['bedrooms'], axis=1)
combine[0] = combine[0].drop(['city'], axis=1)
combine[0] = combine[0].drop(['bed_type'], axis=1)
combine[0] = combine[0].drop(['instant_bookable'], axis=1)
combine[0] = combine[0].drop(['beds'], axis=1)



combine[1] = combine[1].drop(['amenities'], axis=1)
combine[1] = combine[1].drop(['name'], axis=1)
combine[1] = combine[1].drop(['neighbourhood'], axis=1)
combine[1] = combine[1].drop(['description'], axis=1)
combine[1] = combine[1].drop(['thumbnail_url'], axis=1)
combine[1] = combine[1].drop(['latitude'], axis=1)
combine[1] = combine[1].drop(['longitude'], axis=1)
combine[1] = combine[1].drop(['host_has_profile_pic'], axis=1)
combine[1] = combine[1].drop(['host_identity_verified'], axis=1)
combine[1] = combine[1].drop(['first_review'], axis=1)
combine[1] = combine[1].drop(['host_since'], axis=1)
combine[1] = combine[1].drop(['last_review'], axis=1)
combine[1] = combine[1].drop(['cleaning_fee'], axis=1)
combine[1] = combine[1].drop(['bathrooms'], axis=1)
combine[1] = combine[1].drop(['bedrooms'], axis=1)
combine[1] = combine[1].drop(['city'], axis=1)
combine[1] = combine[1].drop(['bed_type'], axis=1)
combine[1] = combine[1].drop(['instant_bookable'], axis=1)
combine[1] = combine[1].drop(['beds'], axis=1)



# print(combine[1]['city'].isnull().values.any())

# combine[0] = pd.concat([combine[0], pd.read_pickle('amenities_train.pickle')], axis=1)
# combine[1] = pd.concat([combine[1], pd.read_pickle('amenities_test.pickle')], axis=1)
#
# combine[0] = pd.concat([combine[0], pd.read_pickle('name_train.pickle')], axis=1)
# combine[1] = pd.concat([combine[1], pd.read_pickle('name_test.pickle')], axis=1)
#
# combine[0] = pd.concat([combine[0], pd.read_pickle('neigh_train.pickle')], axis=1)
# combine[1] = pd.concat([combine[1], pd.read_pickle('neigh_test.pickle')], axis=1)
#
# combine[0] = pd.concat([combine[0], pd.read_pickle('des_train.pickle')], axis=1)
# combine[1] = pd.concat([combine[1], pd.read_pickle('des_test.pickle')], axis=1)


# Remove outliers
combine[0] = combine[0][np.abs(combine[0]['log_price']-combine[0]['log_price'].mean())<=(3*combine[0]['log_price'].std())]

print("Data prep complete.")

print(combine[0].shape)
print(combine[1].shape)

# Training
y_all = combine[0]['log_price']
combine[0] = combine[0].drop("log_price", axis=1)

# frac_test = 0.1
# x_train, x_test, y_train, y_test = train_test_split(combine[0], y_all, test_size = frac_test, random_state = 1987)

x_train = combine[0]
y_train = y_all
x_test = combine[1].drop(['id'], axis=1)

print("Training...")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
# Random Forest
rf = RandomForestRegressor(n_estimators=150, random_state=25)
# rf.fit(x_train, y_train)
# pred = rf.predict(x_test)
# print("Random Forest: ", np.sqrt(mean_squared_error(y_test, pred)))


#Neural Net
# from sklearn.neural_network import MLPRegressor
# mlp = MLPRegressor(hidden_layer_sizes=(15,7,4), random_state = 1984)
# mlp.fit(x_train, y_train)
# pred = mlp.predict(x_test)
# print("Neural Net: ", np.sqrt(mean_squared_error(y_test, pred)))


#Gradient boosting regressor
gbc = GradientBoostingRegressor(n_estimators=100, random_state=2013)
# gbc.fit(x_train, y_train)
# pred = gbc.predict(x_test)
# print("Gradient Boosting: ", np.sqrt(mean_squared_error(y_test, pred)))

# ExtraTrees Regressor
etr = ExtraTreesRegressor(n_estimators=100, random_state=2016)
# etr.fit(x_train, y_train)
# pred = etr.predict(x_test)
# print("Extra Trees: ", np.sqrt(mean_squared_error(y_test, pred)))


# XGBoost
import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb


xg = xgb.XGBRegressor(n_estimators=100, random_state = 1944)
# xg.fit(x_train, y_train)
# pred = xg.predict(x_test)
# print("XGBoost: ", np.sqrt(mean_squared_error(y_test, pred)))

xg_stack = xgb.XGBRegressor(n_estimators=50, random_state = 69)

models = [rf, gbc, etr, xg]

# Use a simple stacking ensemble from https://github.com/vecxoz/vecstack
from vecstack import stacking

s_train, s_test = stacking(models, x_train, y_train, x_test,
    regression = True, metric=mean_squared_error, n_folds = 5, shuffle = True, random_state = 2018, verbose = 2)

xg_stack.fit(s_train, y_train)

final_pred = xg_stack.predict(s_test)
# print("Stacked Ensemble: ", np.sqrt(mean_squared_error(y_test, pred)))


# importances = rf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in rf.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
#
# # # Print the feature ranking
# print("Feature ranking:")
#
# for f in range(x_train.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
# # Plot the feature importances of the forest
#
# plt.figure(1, figsize=(14, 13))
# plt.title("Feature importances")
# plt.bar(range(x_train.shape[1]), importances[indices],
#        color="g", yerr=std[indices], align="center")
# plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
# plt.xlim([-1, x_train.shape[1]])
# plt.savefig("test.png",bbox_inches='tight')
# plt.show()


# Final Predictions
# x_test = combine[1].drop(['id'], axis=1)
# final_pred = rf.predict(x_test)
sub = pd.DataFrame({'id': combine[1]['id'], 'log_price': final_pred})
sub.to_csv("AirBnB Predictions.csv", index=False)













