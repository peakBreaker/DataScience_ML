### Getting started with a ml project - a case study

Imagine we want to do a machine learning project. Our goal is to build
a machine learning model which can classify and regress incoming data based on
supervised ML. The steps are often similar for other problems.

### EDA

First things first, we take a look at our data.  EDA is always the
startingpoint to any machine learning or data science project.  EDA consists
of Graphical and Quantitative data analysis.

Say we have a dataframe with data.  We want to examine it:

```python
df.info()
df.describe()
df.columns
df.dtypes
df.head()
df.tail()
```

Then on we can take a closer look at the columns, say for example we want to
examine number of unique labels.

```python
num_unique_labels = df[LABELS].apply(pd.Series.nunique)
num_unique_labels.plot(kind='bar')

# Label the axes
plt.xlabel('Labels')
plt.ylabel('Number of unique values')

# Display the plot
plt.show()
```

It can also be useful to plot histograms of the data to look at the
distribution, and use ECDF.

### Cleaning the data

In this case study, our data is full of strings. These are encoded as `object`.
Machine learning models perform much better, and it is often required to use
some kind of numerical datatype as our data, thus these object columns must be
converted before we proceed. There are different ways of doing this:

*Category types*

```python
# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
df[LABELS] = df[LABELS].apply(categorize_label)
```

*Dummies*
```python
dummies = pd.get_dummies(df[LABELS], prefix_sep='_')

dummies.head(2)
    label_a  label_b
0         1        0
1         0        1
```
Then we would usually proceed to drop one of the columns to avoid data
duplication.

In our project here, we are turning all our object columns into categorical
columns.

### Measuring success

Once our data looks good for analysis and we have a feel for how it is
distributed, we can make some decision on the loss function for our model.

Since we are trying to solve a classification problem, and we want our model to
be penalized heavily on being confident on wrong guesses, we choose the log
loss function as our loss function

```python
def compute_log_loss(predicted, actual, eps=1e-14):
    """ Computes the logarithmic loss between predicted and
    actual when these are 1D arrays.

    :param predicted: The predicted probabilities as floats between 0-1
    :param actual: the actual binary labels. Either 0 og 1
    :param eps (optional): log(0) is inf, so we need to offset our
                           predicted values slightly by eps from 0 or 1.
    """
    predicted = np.clip(predicted, eps, 1 - eps)
    loss = -1 * np.mean(actual * np.log(predicted)
              + (1 - actual)
              * np.log(1 - predicted))
    return loss
```

### Lets build our model

Its a good idea to begin with a very simple model, to weed out the risky stuff
and get a feel for how challanging this problem will be to model.  In other
words we want to go from raw data to predictions very quickly, instead of being
bogged down in an over complex model from the start.

We'll train our simple model on each label seperatly and use those to predict,
when we'll run it through our loss function and see how well we score.

#### Preparing our data for training

We can use the traditional train_test_split, however since we have categorical
data, we may end up missing categories in our training data. Thus for this
problem using a `StratifiedShuffleSplit` is more appropriate.  It however has
to have one target variable, and our problem has multiple target vars, so we
create a utility function.  It can be found in the multilabel.py file in this
repo.


```python
data_to_train = df[NUMERICAL_COLUMNS].fillna(-1000)
targets = pd.get_dummies(df[LABELS])
X_train, X_test, y_train, y_test = multilabel_train_test_split(
                                   data_to_train, labels_to_use,
                                   size=0.2, seed=123)
```

Great, we now have some train and test data!

#### Training our simple model

Lets use a simple `LogisticRegression` model on our data to begin with. We use
the `OneVsRestClassifier` strategy to fit our model over multiclass
classification:

```python
# Train our model
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)

# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))
```

This gives us an accuracy of 0.0, which is the lowest possible accuracy.  Not
great, but hey - we only used the numeric data.  Lets use categorical data
later, now lets submit our scores on the holdout sets to see the whole workflow
in motion

#### Submitting our score

If this were a competition, we would now load the holdout data, predict on it,
and submit a valid csv file with predictions.  Lets see this in action

```python
holdout = pd.read_csv('Holdout.csv', index_col=0).fillna(-1000)

predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS]).fillna(-1000)

prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns,
                             index=holdout.index,
                             data=predictions)
predictions_df.to_csv('predictions.csv')
score = score_submission(pred_path='predictions.csv')
```

Great, we have now made our first submission.  We can now move onto increasing
our score!

### Introducting NLP

We'll now cover a few techniques for doing Natural Language Processing. Data
for NLP is in the form of text, documents, speech etc.  This is necessary to
give our model the ability to understand the text in our data.  The very basic
concepts in NLP we'll introduce here are as following:

- Tokenization: 
    * splitting strings into segments, segments are lists
    * 'Natural Language Processing' => ['Natural', 'Language', 'Processing']
    * Its basically string splitting.  We can split on whitespace and/or
      special chars
- Representation:
    * Bag of words representation:
        - We can just count the number of times words are occuring
        - But we loose informaton on word order in this technique
    * N-grams:
        - Tokens are 1-grams
        - 2-grams are cols for each word ordered pair.  For example 
          ['Natural Language', 'Language Processing']
        - Goes on for further grams

Lets get started with a bag of words approach on our data.  Naturally
Scikit-Learn got us covered with the `CountVectorizer`

```python
from sklearn.feature_extraction.text import CountVectorizer

TOKENS_BASIC = '\\S+(?=\\s+)'  # Tokenization regex
# TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

df.Program_Description.fillna('', inplace=True)

vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)

vec_basic.fit(df.Program_Description)

print('Num tokens in Program_Description using only whitespace : {}'
      .format(len(vec_basic.get_feature_names)))
```


Lets create a helperfunction to combine all our text data so we can more easily
work with it:

```python
# Define combine_text_columns()
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """

    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)

    # Replace nans with blanks
    text_data.fillna('', inplace=True)

    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)
```

Using this function should give us a series with all the text data, which we
can run through our representation.

```python
text_vector = combine_text_columns(df)
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
vec_alphanumeric.fit_transform(text_vector)
print('There are {} tokens in the dataset'
      .format(len(vec_alphanumeric.get_feature_names)))
```

Sweet! Lets integrate it with our pipeline

### Pipelines

scikit pipelines are great for keeping our sequence of procedures tidy. Lets
set up or earlier example with pipelines:

#### Setting it up

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneCsRestClassifier
from sklearn.model_selection import train_test_split

pl = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression()))
     ])

X_train, X_test, y_train, y_test = train_test_split(
                                   sample_df[['numeric']],
                                   pd.get_dummies(sample_df['label']),
                                   random_state=2)
pl.fit(X_train, y_train)
```

Great, that proves it working.  However our real example needs to weed our
missing values, so we add an imputer

```python
from sklearn.preprocessing import Imputer
pl = Pipeline([
     ('imp', Imputer()),
     ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
```

Great, now lets add the text data!

#### Add NLP to the pipeline

Just to see how it integrates, lets do the NLP alone with the logistic
regression

```python
X_train, X_test, y_train, y_test = train_test_split(
                                   sample_df['text'],
                                   pd.get_dummies(sample_df['label']),
                                   random_state=2)
pl = Pipeline([
        ('vec', CountVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
     ])
```

Now we can fit and score it like always. However we dont want to impute text
data and we cant do NLP on numeric data, so we need to intoducte..

#### Combining it with function transformers and feature unions

In order to glue together the pipelines, we use function transformers and
feature unions.  The idea is to make multiple pipelines and combine them
together.


```python
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
X_train, X_test, y_train, y_test = train_test_split(
                                   sample_df[
                                        ['numeric', 'with_missing', 'text']
                                   ],
                                   pd.get_dummies(sample_df['label']),
                                   random_state=2)
# Function transformers gets the data it wants
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']],
                                    validate=False)

# Set up the different pipelines
numeric_pipeline = Pipeline([
        ('selector', get_numeric_data),
        ('imputer', Imputer())
    ])
text_pipeline = Pipeline([
        ('selector', get_text_data),
        ('vectorizer', CountVectorizer())
    ])

# combine the pipelines
pl = Pipeline([
        ('union', FeatureUnion([
            ('numeric', numeric_pipeline),
            ('text', text_pipeline)
        ])),
        ('clf', OneVsRestClassifier(LogisticRegression()))
     ])
```

Actually just running the FunctionTransformer objects .fit_transform method on our df will
select out the columns from the dataframe which it has been configured for!

Anyway, awesome - we now have a solid flow to work on, and can choose a more
effective model.

### Bringing it all together

We have been only using a few columns till now, but our real dataset has much
more to work with.  We'll use all our helperfunctions till now to combine all
text columns, get all numeric columns and use a multilabel split:

```python
LABELS = ['Function', 'Use', 'Sharing', ..]  # Out target variables
NON_LABELS = [c for c in df.columns if c not in LABELS]  # This will be our predictors
# We also have the names of all numeric cols in the NUMERIC_COLUMNS variable

dummy_labels = pd.get_dummies(df['labels'])  # Turn our target vars to numeric
X_train, X_test, y_train, y_test = multilabel_train_test_split(
                                   df[NON_LABELS], dummy_labels,
                                   size=0.2)
# Function transformers
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

# Build our pipeline
pl = Pipeline([
            ('union', FeatureUnion([
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer)
                ]))
            ])
        ),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
```

Great, now we can easily swap different parts of the pipeline out, such as
trying other models etc!
