from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_features(text_columns, use_year, pooled):
    transformers = []
    for col in text_columns:
        transformers.append((col, TfidfVectorizer(
            lowercase=True, strip_accents='unicode', ngram_range=(1, 2),
            min_df=3, max_df=1.0, sublinear_tf=True,
            # Preserve decimal ICD codes as tokens, including one-character words.
            token_pattern=r'(?u)\b\w[\w.]*\b',
            max_features=15000 if col == 'criteria' else 5000), col))
    if pooled:
        transformers.append(('phase', OneHotEncoder(handle_unknown='ignore'), ['phase']))
    if use_year:
        transformers.append(('year', Pipeline([
            ('impute', SimpleImputer(strategy='median', add_indicator=True)),
            ('scale', StandardScaler()),
        ]), ['start_year']))
    return ColumnTransformer(transformers, remainder='drop', sparse_threshold=1.0)
