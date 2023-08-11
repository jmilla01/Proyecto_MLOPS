<p align=center><img src=https://neurona-ba.com/wp-content/uploads/2021/07/HenryLogo.jpg><p>

# <h1 align=center> **PROYECTO INDIVIDUAL MACHINE LEARNING OPERATIONS (MLOPS)** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>







<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

# JOAQUIN MILLAN LANHOZO - AGOSTO 2023 - DTPT02

# SOY HENRY

El trabajo aqui expuesto fue realizado durante la cursada en la institucion Soy Henry, formo parte de la cohorte DATAPT02 y este es el primer proyecto integrador enfocado en Machine Learning Operations
<hr>  

# ARCHIVOS EN EL REPOSITORIO

<hr> 

+ carpeta "env" --> Tiene el entorno virtual generado para este proyecto.
+ "duracion_peliculas.csv" --> Archivo especifico generado para la consulta de la funcion "peliculas_duracion"
+ "peliculas_paises.csv" --> Csv generado para la funcion de "peliculas_pais"
+ "productoras_exitosas.csv" --> Sub dataset confeccionado para la consulta de la funcion de "productoras_exitosas"
+ countries_counts.csv --> Este csv cuenta con informacion del conteo de las peliculas realizadas por los paises. Funcion "peliculas_pais"
+ "directores.csv" --> Dataframe exportado a csv realizado para la funcion de "get_director"
+ "franquicias.csv" --> Csv elaborado con la informacion necesaria para la funcion "franquicia"
+ "main.py" --> Archivo en donde corren las funciones y por donde realice el deploy a FastApi y disponibilzado luego por Render.
+ "merge_franquicias.csv" --> Csv elaborado con la informacion necesaria para la funcion "franquicia"
+ "productoras_exitosas_df.csv" --> Sub dataset confeccionado para la consulta de la funcion de "productoras_exitosas"
+ "requirements.txt" --> Archivo con las librerias utilizadas y sus versiones.



## Contexto

Para este proyecto, simulamos ser Data Scientist para una start-up en la industria del Streaming. La start-up aun no cuenta con una plataforma desarrollada para poder obtener informacion respecto de las peliculas con las que cuenta la empresa. 

A lo largo del proyecto, se realizaran tareas de ETL (Extraccion, Transformacion y Carga de datos) se extraera informacion de los datasets propuestos, se realizaran algunas transformaciones a algunas columnas para poder tener dataframes mas robustos y completos para poder realizar el EDA (Analisis Exploratorio de Datos) que es un analisis de los datos para poder obtener informacion util.

Luego, realizaremos un modelo de machine learning para poder dar recomendaciones de peliculas y finalmente, el desarrollo de una aplicacion para poder disponibilizar esos datos para que los usuarios puedan consumirlos.


# Proyecto

+ Como primer paso, realizamos la lectura de los datasets propuestos, son dos datasets: uno de "movies", con informacion de las peliculas y otro de "credits" con informacion del elenco que realizaron las peliculas. Los datasets originalmente cuentan con 45466 peliculas

+ Transformaciones:  La data que contenian los datasets no son perfectos, es por eso que se realizan transformaciones para poder utilizar esa data. Algunas columnas fueron transformadas, ya que contenian datos con formatos no adecuados y tambien datos agrupados, que habia que desanidar para poder disponibilizar el dato que contenian esas columnas. Tambien se eliminaron columnas inutiles, que no seran consideradas para el proposito de este proyecto.

+ ** Tratamientos de valores faltantes**: Hay algunos valores nulos a los cuales se realizo un tratamiento de caso que se haya considerado necesario

## Creacion de Funciones
  Se crearon dataframes especificos y mas concretos para eficientizar y acotar el uso de las siguientes funciones. Estas funciones serviran para poder consultar informacion de las peliculas.

+ **peliculas_idioma( *`Idioma`: str* )**:
    Funcion en la cual se ingresa un idioma y retorna la cantidad de películas producidas en ese idioma.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *`X` cantidad de películas fueron estrenadas en `idioma`*
         

+ **peliculas_duracion( *`Pelicula`: str* )**:
    Funcion a la cual se le ingresa una pelicula y devuelve la duracion y el año de estreno.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *`X` . Duración: `x`. Año: `xx`*

+ **franquicia( *`Franquicia`: str* )**:
    Esta funcion recibe un input de la franquicia, y retorna la cantidad de peliculas, ganancia total de la franquicia y el promedio de ganancias
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *La franquicia `X` posee `X` peliculas, una ganancia total de `x` y una ganancia promedio de `xx`*

+ **peliculas_pais( *`Pais`: str* )**:
    La funcion al ingresarle un país , devuelva la cantidad de peliculas que fueron producidas en el mismo.
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *Se produjeron `X` películas en el país `X`*

+ **productoras_exitosas( *`Productora`: str* )**:
    Esta funcion recibe el nombre de una productora, y entrega la ganancia total y la cantidad de peliculas que realizo. 
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *La productora `X` ha tenido un revenue de `x`*

+ **get_director( *`nombre_director`* )**:
    Al ingresar en esta funcion el nombre de un director que se encuentre en el dataset devuelve el éxito del mismo medido a través del retorno. Además, devuelve el nombre de cada película dirigida con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.

+ **Sistema de recomendación**: 
    A esta funcion, se ingresa el nombre de una película y recomienda 5 peliculas similares.

# EDA (Analisis Exploratorio de Datos)

Se realizo un analisis exhaustivo de los datos, en el cual podemos ver datos sobre los idiomas hablados en las peliculas, informacion respecto de los paises productores, los lenguajes originales de las peliculas. las compañias productoras,  colecciones de peliculas, las fechas de estreno, los presupuestos, ganancias, puntajes y valoraciones, generos, actores y directores. 

El dataset acompaña y refleja muy bien, a lo que es la historia del cine. Hitos historicos, sociales y tecnologicas contribuyeron al desarrollo de la industria. 

El 28 de diciembre de 1895, los hermanos Lumière proyectaron una serie de cortometrajes en París, lo que se considera una de las primeras proyecciones públicas de películas en movimiento. Esta fecha a menudo se toma como el comienzo oficial del cine.

Luego, vino el Technicolor en las décadas de 1910 y 1920 lo que llevó al uso más generalizado y a la mejora en la calidad del color en el cine.

Durante esos años posteriores, hasta la decada del 1930, las peliculas eran en blanco y negro. A partir de la decada del 30, acompañado de ciertos hitos las peliculas comenzaron a desarrollarse en mayor cantidad. Estos hitos fueron: El sonido y el habla (anteriormente sin sonido), nacimiento de los musicales, "estreno" de las estrellas de cine y glamour, expansion de generos. Además del sonido, hubo avances técnicos en áreas como la cinematografía, la iluminación y la edición. Estos avances permitieron una mayor calidad visual y narrativa en las películas. Lo que resultó en una mayor popularidad, diversidad y sofisticación de las películas.

Posterior le siguieron las peliculas animadas en los años 40. Snow White and the Seven Dwarfs (1937), esta película,  producida por Walt Disney, es considerada la primera película animada en color y uno de los primeros largometrajes animados. Fue un gran hito en la historia de la animación.

Luego en 1960, tambien se nota otra marcada tendencia del incremento de peliculas producidas. Esto se debe a ciertos factores:

- Surgió lo que se conoció como el "Nuevo Hollywood", donde directores jóvenes y ambiciosos comenzaron a desafiar las convenciones y a explorar temas más oscuros y realistas.

- A nivel global, surgieron movimientos cinematográficos influyentes, como la Nouvelle Vague en Francia y el Neorrealismo italiano. Estos movimientos buscaban formas frescas y auténticas de narrar historias en la pantalla.

-  A medida que las películas se volvían más accesibles y la globalización avanzaba, el cine de diferentes países y culturas ganó popularidad en todo el mundo.

Tambien hay otro crecimiento exponencial marcado en los años 80, ya que las innovaciones tecnológicas, como el uso de efectos especiales y el formato widescreen, permitieron nuevas posibilidades visuales en el cine. A finales de la década de 1970 y principios de la década de 1980, surgieron películas de gran presupuesto con un enfoque en el entretenimiento a gran escala. "Star Wars" (1977) y "Jaws" (1975) son ejemplos destacados de películas que dieron lugar al cine de blockbusters.

 A medida que las películas se volvían más accesibles y la globalización avanzaba, el cine de diferentes países y culturas ganó popularidad en todo el mundo.

 A partir de la década de 1990, sobre todo de los 2000 en adelante,  la tecnología digital comenzó a revolucionar la producción y distribución de películas. El auge de Internet también cambió la forma en que las películas se promocionaban y se distribuían.Acompañado a ello vino el apogeo de un monton de peliculas junto a avances tecnologicos y desarrollo exponencial de la industria cinematografica.


# Deployment

Para finalizar, se realiza la disponibilizacion en un servicio web para que pueda ser consumida por los usuarios 
+ [API](https://ejemplo-joaquinmillan-deploy.onrender.com/docs)

# Fuente de datos

+ [Dataset](https://drive.google.com/drive/folders/1mfUVyP3jS-UMdKHERknkQ4gaCRCO2e1v): Carpeta con los 2 archivos con datos que requieren ser procesados (movies_dataset.csv y credits.csv), tengan en cuenta que hay datos que estan anidados (un diccionario o una lista como valores en la fila).
+ [Diccionario de datos](https://docs.google.com/spreadsheets/d/1QkHH5er-74Bpk122tJxy_0D49pJMIwKLurByOfmxzho/edit#gid=0): Diccionario con algunas descripciones de las columnas disponibles en el dataset.

# Video Deploy
+ [Video](https://drive.google.com/file/d/1uGitaE-bxqTLBstm2RpZiYsGZ_pokkSk/view?usp=drive_link) Video con el deploy y la prueba de las funciones.

# Stack tecnologico

+  Notebook desarrollada en python a traves de Visual Studio Code.
+  Libreria numpy -- [Numpy](https://numpy.org/) Numerical Pyhton útil para realizar cálculos lógicos y matemáticos sobre cuadros y matrices.
+  Libreria pandas -- [Pandas](https://pandas.pydata.org/) Pandas es una librería de Python especializada en el manejo y análisis de estructuras de datos
+  Libreria matplotlib -- [Matplotlib](https://matplotlib.org/) Matplotlib libreria para la visualizacion de datos.
+  Libreria seaborn -- [Seaborn](https://seaborn.pydata.org/) Seaborn libreria para la visualizacion de datos.
+  Libreria datetime -- [Datetime](https://docs.python.org/es/3/library/datetime.html) Datetime libreria utilizada para la transformacion a formato fecha
+  Libreria missingno -- [Missingno](https://pypi.org/project/missingno/) Miisingno visualizacion efectiva de datos nulos
+  Libreria ast -- [ast](https://docs.python.org/3/library/ast.html) El módulo ast ayuda a las aplicaciones de Python a procesar árboles de la gramática de sintaxis abstracta de Python.
+  Libreria sklearn -- [sklearn](https://scikit-learn.org/stable/) Sklearn libreria utilizada para el modelo de Machine Learning
+  Libreria plotly [Plotly](https://plotly.com/python/) Plotly libreria para la visualizacion de datos.
+  FastApi [FastApi](https://fastapi.tiangolo.com/) FastAPI es un web framework para la creacion de APIs con Python 3.7
+  Render [Render](https://render.com/) Utilzacion de Render para el deploy de la API

