import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 0)

train_df = pd.read_csv('data/train.csv')
res_df = pd.read_csv('data/resources_grouped.csv')

print(res_df.head())
train_df = pd.merge(train_df, res_df, on='id', how='left')
print(train_df)
# reduce training data (total # of rows: 1'081'830)
train_df = train_df[:100000]


# 1) Exploration

def show_approve_rate(subset=train_df, desc=''):
    approved = len(subset[subset['project_is_approved'] == 1])
    total = len(subset)
    print('{}: Approval rate {:.1f}% ({}/{})'.format(desc, approved / total * 100, approved, total))


# facet = sns.FacetGrid(train_df, aspect=3, row=None, col=None, hue='project_is_approved')
# facet.map(sns.kdeplot, 'total_price', shade=True)
# facet.set(xlim=(0, train_df['total_price'].max()))
# facet.add_legend()

# sns.barplot(data=train_df, x='teacher_prefix', y="project_is_approved", hue=None)

# show_approve_rate()

# teacher_prefix, school_state, project_submitted_datetime, project_grade_category, project_subject_categories,
# project_subject_subcategories, project_title, project_essay_1, ..., project_essay_4,
# project_resource_summary, teacher_number_of_previously_posted_projects, project_is_approved
# from resources.csv: description, quantity, price

# print(train_df.teacher_prefix.unique())
#
# print(list(train_df.columns.values))

cat_cols = ['teacher_prefix', 'school_state', 'project_grade_category', 'project_subject_categories',
            'project_subject_subcategories']
word_cols = ['project_title', 'essay_students', 'essay_project', 'project_resource_summary', 'description']
other_drop_cols = ['id', 'teacher_id', 'project_submitted_datetime', 'project_essay_1', 'project_essay_2',
                   'project_essay_3', 'project_essay_4', 'price']

drop_cols = cat_cols + word_cols + other_drop_cols

word_feature_nums = [100, 500, 500, 100, 100]
# word_feature_nums = [10, 10, 10, 10, 10]

# 2) Feature engineering
train_df = train_df.fillna(value={'project_title': ''})
train_df = train_df.fillna(value={'project_resource_summary': ''})
train_df = train_df.fillna(value={'description': ''})


def clean(s):
    return re.sub('[^!?.,\w\s]|\x85', '', s)


def essay_students(row):
    if pd.isnull(row.project_essay_3):
        return clean(str(row['project_essay_1']))
    return clean(str(row['project_essay_1']) + ' ' + str(row['project_essay_2']))


def essay_project(row):
    if pd.isnull(row.project_essay_3):
        return clean(str(row['project_essay_2']))
    return clean(str(row['project_essay_3']) + ' ' + str(row['project_essay_4']))


train_df['essay_students'] = train_df.apply(essay_students, axis=1)
train_df['essay_project'] = train_df.apply(essay_project, axis=1)

train_df['description'] = train_df.description.map(clean)
train_df['project_title_len'] = train_df['project_title'].apply(len)
train_df['essay_students_len'] = train_df['essay_students'].apply(len)
train_df['essay_project_len'] = train_df['essay_project'].apply(len)
train_df['project_resource_summary_len'] = train_df['project_resource_summary'].apply(len)
train_df['description_len'] = train_df['description'].apply(len)

train_df['total_price'] = train_df['price'] * train_df['quantity']

train_df = train_df.fillna(value={'teacher_prefix': 'Mrs.'})

dt = pd.to_datetime(train_df['project_submitted_datetime'])
train_df['sub_year'] = pd.DatetimeIndex(train_df['project_submitted_datetime']).year
train_df['sub_month'] = pd.DatetimeIndex(train_df['project_submitted_datetime']).month
train_df['sub_day'] = pd.DatetimeIndex(train_df['project_submitted_datetime']).day
train_df['sub_dayofweek'] = pd.DatetimeIndex(train_df['project_submitted_datetime']).dayofweek
train_df['sub_hour'] = pd.DatetimeIndex(train_df['project_submitted_datetime']).hour

# ax = sns.countplot(x="teacher_number_of_previously_posted_projects", hue="project_is_approved", data=train_df)


# for key, group in train_df.groupby('total_price'):
#     show_approve_rate(group, key)

for col in cat_cols:
    dummies = pd.get_dummies(train_df[col], prefix=col, drop_first=False)
    train_df = pd.concat([train_df, dummies], axis=1)

print(train_df.head())

# 3) Cleaning


# for col in train_df.columns:
#     nans = len(train_df[train_df[col].isnull()])
#     if nans > 0:
#         print('column {} has {} missing values'.format(col, nans))

# column teacher_prefix has 11 missing values
# column project_essay_3 has 1043673 missing values
# column project_essay_4 has 1043673 missing values
# column description has 192 missing values

# print(train_df[train_df['teacher_prefix'].isnull()])

# colormap = plt.cm.RdBu
# plt.figure(figsize=(30, 30))
# plt.title('Pearson Correlation of Features', y=1.05, size=25)
# sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=2.0,
#             square=True, cmap=colormap, linecolor='white', annot=True, fmt='.2f')

# plt.show()

X = train_df.drop(['project_is_approved'], axis=1)
y = train_df['project_is_approved']

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)

for i, col in tqdm(enumerate(word_cols)):
    vectorizer = TfidfVectorizer(max_features=word_feature_nums[i], min_df=3)
    # training_data = vectorizer.fit_transform(X_train[word_cols[i]])
    training_data = np.array(vectorizer.fit_transform(X_train[word_cols[i]]).todense(), dtype=np.float16)
    # val_data = vectorizer.transform(X_val[word_cols[i]])
    val_data = np.array(vectorizer.fit_transform(X_val[word_cols[i]]).todense(), dtype=np.float16)

    for j in range(word_feature_nums[i]):
        X_train[col + '_' + str(j)] = training_data[:, j]
        X_val[col + '_' + str(j)] = val_data[:, j]
    # Falls man sich die Matrix anschauen will, geht das so:
    # frequency_matrix = pd.DataFrame(data=training_data.toarray(), columns=vectorizer.get_feature_names())

print(X_train.head())

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 12,
    'num_leaves': 31,  # 127 ist nicht besser
    'learning_rate': 0.025,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'verbose': 0,
    'num_threads': 1,
    'lambda_l2': 1,
    'min_gain_to_split': 0,
}

X_train = X_train.drop(drop_cols, axis=1)
X_val = X_val.drop(drop_cols, axis=1)

feature_names = list(X_train.columns)
print('{} features: {}'.format(len(feature_names), feature_names))

model = lgb.train(
    params,
    lgb.Dataset(X_train, y_train, feature_name=feature_names),
    num_boost_round=1000,
    valid_sets=[lgb.Dataset(X_val, y_val)],
    early_stopping_rounds=100,
    verbose_eval=100,
)

importance = model.feature_importance()
model_fnames = model.feature_name()
tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
tuples = [x for x in tuples if x[1] > 0]
print('Important features:')
print(tuples[:50])

p = model.predict(X_val, num_iteration=model.best_iteration)
auc = roc_auc_score(y_val, p)
print('AUC: {}'.format(auc))

# test data
test_df = pd.read_csv('data/test.csv')
test_ids = test_df['id'].values
test_df = pd.merge(test_df, res_df, on='id', how='left')

test_df = test_df.fillna(value={'project_title': ''})
test_df = test_df.fillna(value={'project_resource_summary': ''})
test_df = test_df.fillna(value={'description': ''})

test_df['essay_students'] = test_df.apply(essay_students, axis=1)
test_df['essay_project'] = test_df.apply(essay_project, axis=1)

test_df['description'] = test_df.description.map(clean)
test_df['project_title_len'] = test_df['project_title'].apply(len)
test_df['essay_students_len'] = test_df['essay_students'].apply(len)
test_df['essay_project_len'] = test_df['essay_project'].apply(len)
test_df['project_resource_summary_len'] = test_df['project_resource_summary'].apply(len)
test_df['description_len'] = test_df['description'].apply(len)

test_df['total_price'] = test_df['price'] * test_df['quantity']

test_df = test_df.fillna(value={'teacher_prefix': 'Mrs.'})

dt = pd.to_datetime(test_df['project_submitted_datetime'])
test_df['sub_year'] = pd.DatetimeIndex(test_df['project_submitted_datetime']).year
test_df['sub_month'] = pd.DatetimeIndex(test_df['project_submitted_datetime']).month
test_df['sub_day'] = pd.DatetimeIndex(test_df['project_submitted_datetime']).day
test_df['sub_dayofweek'] = pd.DatetimeIndex(test_df['project_submitted_datetime']).dayofweek
test_df['sub_hour'] = pd.DatetimeIndex(test_df['project_submitted_datetime']).hour

for col in cat_cols:
    dummies = pd.get_dummies(test_df[col], prefix=col, drop_first=False)
    test_df = pd.concat([test_df, dummies], axis=1)

X_test = test_df.drop(drop_cols, axis=1)
preds = model.predict(X_test, num_iteration=model.best_iteration)

subm = pd.DataFrame()
subm['id'] = test_ids
subm['project_is_approved'] = preds
subm.to_csv('submission.csv', index=False)
