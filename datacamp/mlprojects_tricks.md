### Tips and tricks in mlprojects

Some routines are worth getting in there, lets follow some tuts on tips and
tricks from the experts.  The tips involve:

- Text processing 
- Stat methods
- Computation efficiency

#### Text tricks
- Tokenization on punctuation
- Include different N-grams

```python
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
vec = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC, ngram_range=(1,2))
```

#### Stat tricks

- interaction terms:
    * How important is it that features come together?
    * lets say we have a linear model 
        `beta*x1 + beta*x2 + beta3*(x1*x2)`
        The interactionterm is the last part of the function, as it only has
        value when both x1 and x2 have a value
    * sklearn.preprocessing.PolinomialFeatures
    * Can also use sklearn.preproccesing.SparseInteraction
    * Acts as a transform on the data, so its easy to add near the end before
      the actual model
- Hashing tricks
    * To make computation more feasable and computationally efficient, we want
      to make our array of features as small as possible
    * Hashing means turning values into hashes, which are of a fixed size, 
      which can be very useful in increasing computation efficiency
    * Just switch out the CountVectorizer with the HashVectorizer in sklearn

#### Winning model

Heres an example of a kaggle winning model based on logistic regression:

```python
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                     non_negative=True, norm=None, binary=False,
                                                     ngram_range=(1,2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
```
