import requests
import random

def get_toponym_pop(toponym, users_list):
    current_user = users_list[random.randint(0, 4)]
    url = "http://api.geonames.org/searchJSON?q="
    url += toponym + "&orderby=population&maxRows=1&username="+current_user
    r = requests.get(url)
    return r.json()

