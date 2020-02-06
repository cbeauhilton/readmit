from lightgbm import LGBMClassifier
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline

import pandas

df = pandas.read_csv("audit.csv")

cat_columns = ["Education", "Employment", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

mapper = DataFrameMapper(
	[([cat_column], [CategoricalDomain(), LabelBinarizer()]) for cat_column in cat_columns] +
	[(cont_columns, ContinuousDomain())]
)
classifier = LGBMClassifier(objective = "binary", n_estimators = 31, random_state = 42)

pipeline = PMMLPipeline([
	("mapper", mapper),
	("classifier", classifier)
])
pipeline.fit(df, df["Adjusted"])

sklearn2pmml(pipeline, "LightGBMAudit.pmml") 



from sklearn.preprocessing import LabelBinarizer

mapper = DataFrameMapper(
	[([cat_column], [CategoricalDomain(), LabelBinarizer()]) for cat_column in cat_columns] +
	[(cont_columns, ContinuousDomain())]
)

# A categorical feature transforms to a variable number of binary features.
# One way of obtaining a list of binary feature indices is to transform the dataset, and exclude the indices of known continuous features
Xt = mapper.fit_transform(df)
cat_indices = [i for i in range(0, Xt.shape[1] - len(cont_columns))]

pipeline = PMMLPipeline([...])
pipeline.fit(df, df["Adjusted"], classifier__categorical_feature = cat_indices)




from sklearn.preprocessing import LabelEncoder

mapper = DataFrameMapper(
	[([cat_column], [CategoricalDomain(), LabelEncoder()]) for cat_column in cat_columns] +
	[(cont_columns, ContinuousDomain())]
)

# A categorical string feature transforms to exactly one categorical integer feature.
# The list of categorical feature indices contains len(cat_columns) elements
cat_indices = [i for i in range(0, len(cat_columns))]

pipeline = PMMLPipeline([...])
pipeline.fit(df, df["Adjusted"], classifier__categorical_feature = cat_indices)


from sklearn.impute import SimpleImputer

mapper = DataFrameMapper(
	[([cat_column], [CategoricalDomain(), SimpleImputer(strategy = "most_frequent"), LabelEncoder()]) for cat_column in cat_columns] +
	[(cont_columns, ContinuousDomain())]
)

pipeline = PMMLPipeline([...])
pipeline.fit(df, df["Adjusted"], classifier__categorical_feature = cat_indices)



from sklearn2pmml.preprocessing import PMMLLabelEncoder

mapper = DataFrameMapper(
	[([cat_column], [CategoricalDomain(), PMMLLabelEncoder()]) for cat_column in cat_columns] +
	[(cont_columns, ContinuousDomain())]
)

pipeline = PMMLPipeline([...])
pipeline.fit(df, df["Adjusted"], classifier__categorical_feature = cat_indices)