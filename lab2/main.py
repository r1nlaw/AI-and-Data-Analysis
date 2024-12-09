import nltk
import spacy
import re
from nltk import FreqDist
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer, SnowballStemmer

# Загрузка необходимых ресурсов
nltk.download('punkt')
nltk.download('stopwords')

# Данные
data = "All work and no play makes jack a dull boy, all work and no play"
tokens = re.findall(r'\b\w+\b', data)  # Регулярное выражение для выделения слов
print("Tokens: ", tokens)

# Создаём униграммы и биграммы
unigram = list(nltk.ngrams(tokens, 1))
bigram = list(nltk.ngrams(tokens, 2))

# Частотное распределение
print('Популярные униграммы: ', FreqDist(unigram).most_common(5))
print('Популярные биграммы: ', FreqDist(bigram).most_common(5))

# Выводим первые 5 униграмм и биграмм для проверки
print("Первые 5 униграмм: ", unigram[:5])
print("Первые 5 биграмм: ", bigram[:5])

# Загружаем стоп-слова
stopWords = set(stopwords.words('english'))

# Текст для создания облака слов
some_text = "Impossible is a word to be found only in the dictionary of fools."

# Задаем список стоп-слов
stopWords_custom = ["is", "a", "to", "be", "in", "the", "of", "only"]

# Удаляем стоп-слова из текста
some_text_without_stop_words = some_text
for word in stopWords_custom:
    some_text_without_stop_words = some_text_without_stop_words.replace(f" {word} ", " ")

# Создание облаков слов
wordcloud_with_stop_words = WordCloud(width=800, height=400, background_color='white').generate(some_text)
wordcloud_without_stop_words = WordCloud(width=800, height=400, background_color='white').generate(some_text_without_stop_words)

# Отображение облака слов со стоп-словами
plt.figure(figsize=(10, 5))
plt.title("Облако слов с включением стоп-слов")
plt.imshow(wordcloud_with_stop_words, interpolation='bilinear')
plt.axis('off')
plt.show()

# Отображение облака слов без стоп-слов
plt.figure(figsize=(10, 5))
plt.title("Облако слов без стоп-слов")
plt.imshow(wordcloud_without_stop_words, interpolation='bilinear')
plt.axis('off')
plt.show()

# Применяем стемминг к английским словам
words = ["game", "gaming", "gamed", "games", "compacted"]
ps = PorterStemmer()
print("Stemming for English words: ", list(map(ps.stem, words)))

# Применяем стемминг для русского языка
words_ru = ['корова', 'мальчики', 'мужчины', 'столом', 'убежала']
stemmer_ru = SnowballStemmer("russian")
stemmed_words_ru = [stemmer_ru.stem(word) for word in words_ru]
print("Stemming for Russian words: ", stemmed_words_ru)

# Лемматизация с помощью spaCy
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
is no basis for a system of government.  Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""

nlp = spacy.load("en_core_web_sm")
doc = nlp(raw)

# Лемматизация текста
print("Lemmatized text: ", ' '.join([token.lemma_ for token in doc]))

# Создаем список из первых 7 токенов с их леммами и частями речи
print("First 7 tokens with lemmata and POS: ", [(token.lemma_, token.pos_) for token in doc[:7]])

# Регулярные выражения: извлечение чисел
print("Numbers in text: ", re.findall('\d+', 'There is some numbers: 49 and 432'))

# Регулярные выражения: замена знаков препинания и разбиение текста
print("Text after punctuation removal: ", re.sub('[,\.?!]',' ','How, to? split. text!').split())

# Регулярные выражения: удаление всех символов, кроме букв A-z
print("Text after removing non-alphabetic characters: ", re.sub('[^A-z]',' ','I 123 can 45 play 67 football').split())
