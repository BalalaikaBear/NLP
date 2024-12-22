from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#import nltk
#nltk.download('subjectivity')

text_rus = "NLTK упрощает обработку текста. Разве это не прекрасно?"
text_eng = "The stemmed form of leaves is leaf"

# ТОКЕНИЗАЦИЯ ------------------------------------------------------------------------------------------------------- *
# разбиение на слова
word_tokens: list[str] = word_tokenize(text_rus)
print('WORD TOKENS:', word_tokens)

# разбиение на предложения
sentence_tokens: list[str] = sent_tokenize(text_rus)
print('SENTENCE TOKENS:', sentence_tokens)

# УДАЛЕНИЕ СТОП-СЛОВ ------------------------------------------------------------------------------------------------ *
stop_words: set[str] = set(stopwords.words('russian'))
filtered_tokens: list[str] = [word for word in word_tokens if word not in stop_words]
print('FILTERED TOKENS:', filtered_tokens)

# СТЕММИНГ ---------------------------------------------------------------------------------------------------------- *
# на английском
stemmer: PorterStemmer = PorterStemmer()
tokens: list[str] = word_tokenize(text_eng)
stemmed_words: list[str] = [stemmer.stem(word) for word in tokens]
print('ENGLISH STEMMER:', stemmed_words)

# на русском
stemmer: SnowballStemmer = SnowballStemmer("russian")
tokens: list[str] = word_tokenize(text_rus)
stemmed_words: list[str] = [stemmer.stem(word) for word in tokens]
print('RUSSIAN STEMMER:', stemmed_words)

# ЛЕММАТИЗАЦИЯ ------------------------------------------------------------------------------------------------------ *
# на английском
lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
tokens: list[str] = word_tokenize(text_eng)
lemmatized_words: list[str] = [lemmatizer.lemmatize(word) for word in tokens]
print('ENGLISH LEMMATIZER:', lemmatized_words)

# на русском (отсутствует)
stemmer: SnowballStemmer = SnowballStemmer("russian")
tokens: list[str] = word_tokenize(text_rus)
lemmatized_words: list[str] = [stemmer.stem(word) for word in tokens]
print('RUSSIAN LEMMATIZER:', lemmatized_words)

# АНАЛИЗ НАСТРОЕНИЙ ------------------------------------------------------------------------------------------------- *
# Простая классификация с использованием предварительно обученных данных
sia: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
text = "NLTK is amazing for natural language processing!"
print('SENTIMENT INTENSITY:', sia.polarity_scores(text))

# Анализ настроений с использованием токенизатора и списка стоп-слов
stop_words: set[str] = set(stopwords.words('english'))
filtered_text: str = ' '.join([word for word in word_tokenize(text) if not word in stop_words])
sia: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
print('SENTIMENT INTENSITY with FILTERED TOKENS:', sia.polarity_scores(filtered_text))

# Комбинирование лемматизации и анализа настроений
lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
lemmatized_text: str = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])
sia: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
print('SENTIMENT INTENSITY with LEMMATIZER:', sia.polarity_scores(lemmatized_text))

# МОДЕЛЬ BOW (Bag of Words | BoW) ----------------------------------------------------------------------------------- *
# Создание BoW с NLTK и использование его для классификации
# Пример данных
texts = ["I love this product", "This is a bad product", "I dislike this", "This is the best!"]
labels = [1, 0, 0, 1]  # 1 - позитивный, 0 - негативный

# Токенизация
tokens = [word_tokenize(text) for text in texts]

# Создание BoW модели
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform([' '.join(token) for token in tokens])

# Разделение данных на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(bow, labels, test_size=0.3)

# Обучение классификатора
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Оценка классификатора
predictions = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
