#!/usr/bin/env python
# coding: utf-8

# <center><h1>Inteligencia Artificial</header1></center>

# Presentado por: Johana Salazar  <br>
# Fecha: DD/MM/2022

# # Importación de librerias necesarias

# In[1]:


#Para esta actividad se importarán las siguientes librerías:

#Para esta actividad se importarán las siguientes librerías:
import pandas as pd
import numpy as np
#importar libreria encoding
from sklearn.preprocessing import LabelEncoder
#Librerias para el grafico
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# ## Cargar el Dataset

# In[7]:


#Código para cargar el Dataset

#Cargar el data set insurance
pima = pd.read_csv("Clean_dataset.csv")
#Mostrar los datos
pima.head()


# ## Descripción de la fuente del Dataset

# El objetivo del estudio es analizar el conjunto de datos de reserva de vuelos obtenidos del sitio web "Ease My Trip" y realizar varias pruebas de hipótesis estadísticas para obtener información significativa de él.

# ## Explique el problema a resolver. 
# Descripción del problema. Tipo de problema (justifique). Variable objetivo, variables de entrada. Utilidad de su posible solución. Elementos adicionales que considere relevantes (no son necesarios contenidos teóricos, sino explicar qué relaciones tratas de comprobar y con qué métodos).

# En el siguiente dataset representa los datos que permiten obtener la informacion medica de una compania del seguro.

# ## Caracterización del Dataset
# 
# Realice una descripción de los datos con:
# 
# >- Número de instancias en total.
# >- Número de atributos de entrada, su significado y tipo.
# >- Estadísticas de la variable objetivo.
# >- Estadísticas los atributos en relación con la variable objetivo.
# 

# Las diversas características del conjunto de datos limpio se explican a continuación:
# 
# Línea aérea: el nombre de la compañía aérea se almacena en la columna de la línea aérea. Es una característica categórica que tiene 6 aerolíneas diferentes.
# Vuelo: Vuelo almacena información sobre el código de vuelo del avión. Es una característica categórica.
#  Ciudad Origen: Ciudad desde donde despega el vuelo. Es una característica categórica que tiene 6 ciudades únicas.
#  Hora de salida: esta es una característica categórica derivada que se obtiene al agrupar períodos de tiempo en contenedores. Almacena información sobre la hora de salida y tiene 6 etiquetas de tiempo únicas.
# Paradas: una característica categórica con 3 valores distintos que almacena el número de paradas entre las ciudades de origen y destino.
# Hora de llegada: esta es una característica categórica derivada creada al agrupar intervalos de tiempo en contenedores. Tiene seis etiquetas de tiempo distintas y mantiene información sobre la hora de llegada.
# Ciudad de Destino: Ciudad donde aterrizará el vuelo. Es una característica categórica que tiene 6 ciudades únicas.
# Clase: Una característica categórica que contiene información sobre la clase de asiento; tiene dos valores distintos: Negocios y Economía.
# Duración: una función continua que muestra la cantidad total de tiempo que lleva viajar entre ciudades en horas.
#  Días Restantes: Es una característica derivada que se calcula restando la fecha del viaje a la fecha de la reserva.
#  Precio: La variable objetivo almacena información del precio del boleto.

# In[9]:


#Código que responde a la descripción anterior
pima.info() 


# En un par de párrafos haga un resumen de los principales hallazagos encontrados:    

# ## Preprocesamiento del dataset. Transformaciones previas necesarias para la modelación

# In[10]:


pima.info


# In[ ]:





# In[11]:


#Transformación de las caracteristicas (mire el apartado Feature engineering del aula virtual)
#Se usa labelEncoder para normalizar etiquetas no numericas y en este caso esta el sexo, la region y el smoker.
encoder = LabelEncoder()


# ## División del dataset en datos de entrenamiento y datos de test 

# In[12]:


#Código que realice la división en entrenamiento y test, de acuerdo con la estretgia de evaluación planeada. Describa cuál es.
pima["price"]= encoder.fit_transform(pima["price"])


# In[23]:


pima.head(100)


# In[24]:


pima["source_city"]= encoder.fit_transform(pima["source_city"])
pima.head(100)


# In[25]:


pima["destination_city"]= encoder.fit_transform(pima["destination_city"])
pima.head(100)


# ## Modelamiento

# In[ ]:





# In[14]:


one_hot_encoded_pima = pd.get_dummies(pima, columns = ['source_city', 'destination_city'])
print(one_hot_encoded_pima)


# In[16]:


#Códo del modelo

new_data = pd.get_dummies(data=pima, drop_first=True)
print(new_data)


# In[31]:


#Entrenamiento

lin_reg = linear_model.LinearRegression()


# In[33]:


#Test


# ## Evaluación del Modelo

# Construya un o dos párrafos con los principales hallazgos. Cómo está funcionando el modelo? Calidad en los resultados de predicción. 

# In[ ]:




