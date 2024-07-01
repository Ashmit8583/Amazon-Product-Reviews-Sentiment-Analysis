import pandas as pd
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('amazon_reviews.csv')

# Visualize sentiment distribution
sns.countplot(df['sentiment'])
plt.title('Sentiment Distribution')
plt.show()

# Generate word cloud
text = ' '.join(review for review in df['review_text'])
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Prepare data for model
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review_text'])
y = df['sentiment']

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Evaluate model
y_pred = model.predict(X)
print(classification_report(y, y_pred))
print(confusion_matrix(y, y_pred))
