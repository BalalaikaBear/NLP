from dataclasses import dataclass
from enum import StrEnum, auto


class Type(StrEnum):
    PIZZA = auto()
    DRINK = auto()


class Ingr(StrEnum):
    MOZZARELLA = auto()
    PARMESAN = auto()
    CHEDDAR = auto()
    PEPPERONI = auto()
    HAM = auto()
    CHICKEN = auto()
    BACON = auto()
    SALAMI = auto()
    SAUSAGES = auto()
    BEEF = auto()
    PORK = auto()
    PINEAPPLE = auto()
    SHRIMP = auto()
    TOMATOES = auto()
    ONION = auto()
    BELLPEPPER = auto()
    MUSHROOMS = auto()
    OLIVES = auto()
    EGGPLANT = auto()


@dataclass
class Product:
    name: str
    type: str
    price: int
    products: set[str] = None

    def __str__(self):
        return self.name


products: list[Product] = [
    Product('Barbecue', Type.PIZZA, 599, {Ingr.BEEF, Ingr.PEPPERONI, Ingr.MOZZARELLA, Ingr.TOMATOES}),
    Product('Cheese', Type.PIZZA, 309, {Ingr.MOZZARELLA, Ingr.PARMESAN, Ingr.CHEDDAR}),
    Product('Pepperoni', Type.PIZZA, 369, {Ingr.PEPPERONI, Ingr.MOZZARELLA, Ingr.TOMATOES}),
    Product('Cola', Type.DRINK, 149),
    Product('Apple juice', Type.DRINK, 199),
]

text1: str = 'Заказать пиццу с ананасами и колу'
text2: str = 'Добавь две средних пеперони с луком и бутылку сока'
text_sum: str = text1 + '. ' + text2
text_order = 'Заказ Заказать купить покупать закупить заказывать закажу'

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

tokens: set[str] = word_tokenize(text_order)

stemmer: SnowballStemmer = SnowballStemmer("russian")
lemmatized_words: list[str] = [stemmer.stem(word) for word in tokens]
print('RUSSIAN LEMMATIZER:', lemmatized_words)




import spacy
from spacy.pipeline import EntityRuler
nlp = spacy.load('ru_core_news_sm')

patterns = [
    {'label': Type.PIZZA, 'pattern': [{'lower': {'in': ['моцарелла', 'моцареллы', 'моцареллу']}}]}
]

ruler = EntityRuler(nlp, overwrite_ents=True)
ruler.add_patterns(patterns)
nlp.add_pipe()

#ruler = nlp_app.add_pipe('entity_ruler')
#ruler.add_patterns(patterns)

doc = nlp('двадцать четыре мальчика пошли в Нигерию поесть моцареллы, где их встретил Павел')
print(doc.ents)
