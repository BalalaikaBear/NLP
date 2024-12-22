from django.shortcuts import render
from django.http import HttpRequest, HttpResponse

from nlp.objects import Size, Ingredient, Product, Pizza, Snack, Drink, OrderItem, Order
from nlp.menu import menu

def main(request: HttpRequest) -> HttpResponse:
    """Главная страница"""
    context = {
        'menu': menu
    }

    return render(request, 'main.html')
