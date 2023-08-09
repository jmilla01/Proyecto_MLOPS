from pydantic import BaseModel
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import zipfile

app = FastAPI(title='Proyecto Individual MLOPS ',description='Joaquin Millan Lanhozo')

# Carga de archivos especificos, armaros para las funciones

movies_language = pd.read_csv("C:\Users\tomas\Desktop\DS - HENRY\00.LABS\1.PIMLOPS\GITHUB\movies_language.csv")
movies = pd.read_csv("C:\Users\tomas\Desktop\DS - HENRY\00.LABS\1.PIMLOPS\GITHUB\Duracion_peliculas.csv")
merge_franquicias = pd.read_csv("C:\Users\tomas\Desktop\DS - HENRY\00.LABS\1.PIMLOPS\GITHUB\franquicias.csv")
countries_counts = pd.read_csv("C:\Users\tomas\Desktop\DS - HENRY\00.LABS\1.PIMLOPS\GITHUB\Peliculas_paises.csv")
franquicias = pd.read_csv("C:\Users\tomas\Desktop\DS - HENRY\00.LABS\1.PIMLOPS\GITHUB\franquicias.csv")
directores = pd.read_csv("C:\Users\tomas\Desktop\DS - HENRY\00.LABS\1.PIMLOPS\GITHUB\directores.csv")


@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma:str):
    '''Ingresas el idioma, retornando la cantidad de peliculas producidas en el mismo'''

    # Ante error de ingresar mayuscula, se pasa a minuscula
    idioma = idioma.lower()

    # Pasaje del idioma a str
    idioma = str(idioma)

    # Condicional para que busque y tome la cantidad del idioma dado
    if idioma in movies_language.index:
        count = movies['original_language'].value_counts().get(idioma, 0)
        return {'idioma':idioma, 'cantidad':count}

    #en caso de error, que lo marque la funcion
    else:
        return "Codigo de idioma mal ingresado"
    

@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula:str):
    '''Ingresas la pelicula, retornando la duracion y el año'''

    #manejo de errores
    pelicula = pelicula.lower().title()
    pelicula = str(pelicula)

    # Condicinal para que busque, filtre y traiga la informacion correspondiente al input dado 
    if pelicula in movies["title"].values:
        
        pelicula_info = movies[movies["title"]== pelicula].title.values
        duracion = movies[movies["title"]== pelicula].runtime.values
        anio = movies[movies["title"]== pelicula].release_year.values
        return {'pelicula':pelicula_info, 'duracion':duracion, 'anio':anio}
    # en caso de error, dar mensaje de error
    else:
        return "Lo lamentamos! La pelicula buscada no tiene esta informacion al respecto" 
    
@app.get('/franquicia/{franquicia}')
def franquicias(franquicia:str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''

    # cambio de formato del valor
    franquicia = str(franquicia)

    #condicional para traer los valores de la funcion
    if franquicia in merge_franquicias["Franchise"].values:
        
        name_franquicia = merge_franquicias[merge_franquicias["Franchise"]== franquicia].Franchise.values[0]
        cantidad_de_peliculas_franquicia = merge_franquicias[merge_franquicias["Franchise"]== franquicia].Movie_Count.values[0]
        ganancia_total_franquicia = merge_franquicias[merge_franquicias["Franchise"]== franquicia].revenue.values[0]
        ganancia_promedio_franquicia = merge_franquicias[merge_franquicias["Franchise"]== franquicia].ganancia_promedio_por_franquicia.values[0]
        
        
        return {'franquicia':franquicia, 'cantidad':cantidad_de_peliculas_franquicia, 'ganancia_total':ganancia_total_franquicia, 'ganancia_promedio':ganancia_promedio_franquicia}
    # en caso de error, dar mensaje de error    
    else:
        return "Lo lamentamos! La franquicia buscada no tiene esta informacion al respecto" 


@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais:str):
    '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo'''
    
    Pais = str(Pais)
    if Pais in countries_counts["production_countries_names"].values:
        
        name_pais = countries_counts[countries_counts["production_countries_names"]== Pais].production_countries_names.values
        conteo_paises = countries_counts[countries_counts["production_countries_names"]== Pais].Movie_Count.values

        return {'pais':name_pais, 'cantidad':conteo_paises}

    # en caso de error, dar mensaje de error
    else:

        return "Lo lamentamos! El pais buscado no produjo peliculas"

@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora:str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''
    productora = str(productora)
    if productora in franquicias["Production_Company"].values:
        
        name_franquicia = productoras_exitosas[productoras_exitosas["Production_Company"]== productora].Production_Company.values[0]
        cantidad_de_peliculas_productora = productoras_exitosas[productoras_exitosas["Production_Company"]== productora].Movie_Count.values[0]
        ganancia_total_productora = productoras_exitosas[productoras_exitosas["Production_Company"]== productora].revenue.values[0]
        
        
        return {'franquicia':franquicia, 'cantidad':cantidad_de_peliculas_franquicia, 'revenue_total':ganancia_total_franquicia}
        
    else:
        return "Lo lamentamos! La franquicia buscada no tiene esta informacion al respecto" 



@app.get('/get_director/{nombre_director}')
def get_director(nombre_director):
    """
    Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo 
    medido a través del retorno. Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, 
    retorno individual, costo y ganancia de la misma, en formato lista.
    """ 
    # Filtrar el DataFrame para obtener las películas dirigidas por el director dado
    peliculas_director = directores[directores['director_name'] == nombre_director]
    
    # Verificar si el director se encuentra en el DataFrame
    if peliculas_director.empty:
        return None  # O un mensaje indicativo de que el director no se encontró
    
    # Calcular el éxito del director sumando los retornos individuales de sus películas
    retorno_total_director = peliculas_director['return'].sum()
    
    # Crear una lista de diccionarios con información detallada de cada película
    peliculas = []
    for index, row in peliculas_director.iterrows():
        pelicula_info = {
            'nombre': row['title'],
            'anio': row['release_date'],
            'retorno_pelicula': row['return'],
            'budget_pelicula': row['costo'],
            'revenue_pelicula': row['revenue']
        }
        peliculas.append(pelicula_info)
    
    # Crear el diccionario de respuesta
    respuesta = {
        'director': nombre_director,
        'retorno_total_director': retorno_total_director,
        'peliculas': peliculas
    }
    
    return respuesta
