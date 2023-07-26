#!/usr/bin/env python
# encoding: utf-8

import tweepy  # https://github.com/tweepy/tweepy
import csv

pathClavesTwitter = "/bolsa/twitterapikeys/keys"
pathDestinoTweets = "/bolsa/twitter"


def vaciarCarpeta(folder):
    import os, shutil
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


clavesTwitter = {}
with open(pathClavesTwitter) as myfile:
    for line in myfile:
        name, var = line.partition("=")[::2]
        clavesTwitter[name.strip()] = var.rstrip()

# NO DESCOMENTAR, PORQUE SE VERÍAN LAS CLAVES EN EL LOG!!
# print(clavesTwitter)

# Twitter API credentials
api_key = clavesTwitter.get("apikey")
api_key_secret = clavesTwitter.get("apikeysecret")
access_token = clavesTwitter.get("accesstoken")
access_token_secret = clavesTwitter.get("accesstokensecret")

import os
import tweepy as tw
import pandas as pd

auth = tw.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth)

# USO
api = tweepy.API(auth)

# Verificación de la conexión
api.verify_credentials()


def get_all_tweets(screen_name):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # Se asume que está ya autenticado

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print(f"getting tweets before {oldest}")

        # all subsequent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print(f"...{len(alltweets)} tweets downloaded so far")

    # transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]

    # write the csv
    with open(pathDestinoTweets + "/" + f'{screen_name}' + "_tweets.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "created_at", "text"])
        writer.writerows(outtweets)

    pass


if __name__ == '__main__':

    # Se vacía la carpeta donde se guardarán los tuits
    # vaciarCarpeta(pathDestinoTweets)

    tuiterosRecomendacionAOjo = ["Boytrader08", "StocksThatGo"]
    # Se descarga la info de todos los tuiteros
    for cuenta in tuiterosRecomendacionAOjo:
        # get_all_tweets(cuenta)
        a = 2

    # DEFINICIÓN de FEATURE:
    # Para la empresa XXXX, la feature contará cuántos tuiteros mencionan el ticker $XXXX en el conjunto de los
    # YY días previos a la fila analizada. El resultado será 0, 1, 2, 3...
    # El nombre de la feature será: MENCIONES-TWITTER-YY-DIAS
    fechaRowActual = "2022-01-01"
    empresaRowActual = "AVCT"

    diasPreviosPresenciaEnTuit = [1, 5]

    from datetime import timedelta
    from datetime import datetime as datetime2

    FORMATO_FECHA_TWITTER = '%Y-%m-%d'

    # Para cada periodo de presencia, se creará una feature:
    fechaRowActual_obj = datetime2.strptime(fechaRowActual, FORMATO_FECHA_TWITTER)
    for diaspreviosmaximos in diasPreviosPresenciaEnTuit:
        numeroMencionesTotales = 0

        ficheros = os.listdir(pathDestinoTweets)

        for fichero in ficheros:
            pathFichero = os.path.join(pathDestinoTweets, fichero)
            if os.path.isfile(pathFichero) and fichero.endswith('.csv'):
                numeroMencionesPorElTuitero = 0
                # Cada fichero pertenece a un tuitero
                tuitero = fichero.split("_")[0]
                # Se lee el fichero
                tuits = pd.read_csv(filepath_or_buffer=pathFichero, sep=',')

                # Se recorre cada fila
                for index, fila in tuits.iterrows():
                    fechaTuit = fila['created_at'].split()[0]
                    fechaTuit_obj = datetime2.strptime(fechaTuit, FORMATO_FECHA_TWITTER)

                    fechaAntiguaMaxima_obj = fechaRowActual_obj - timedelta(days=diaspreviosmaximos)
                    fechaAntiguaMaxima = fechaAntiguaMaxima_obj.strftime(FORMATO_FECHA_TWITTER)
                    if fechaAntiguaMaxima_obj <= fechaTuit_obj and fechaTuit_obj <= fechaRowActual_obj:
                        # Aquí sólo habrá filas dentro del rango de fechas aceptado
                        # print("El tuitero " + cuenta + " ha escrito en la fecha " + fila[
                        #     'created_at'] + " el siguiente tuit: " + fila['text'])
                        if ("$" + empresaRowActual) in fila['text']:
                            # Se aumentará el contador de menciones del tuitero sobre la empresa
                            # print("ENCONTRADO: " + fechaTuit + " debe ser posterior a " + fechaAntiguaMaxima)
                            numeroMencionesPorElTuitero += 1

                print("Número de menciones de la empresa " + empresaRowActual +
                      " por el tuitero " + tuitero + " : " + str(numeroMencionesPorElTuitero) + " entre esta fecha: " +
                      fechaRowActual + " y los " + str(diaspreviosmaximos) + " días anteriores")

                if numeroMencionesPorElTuitero > 0:
                    numeroMencionesTotales += 1

        print("Número de tuiteros que referencian al menos una vez a la empresa " + empresaRowActual + " en los "
              + str(diaspreviosmaximos) + " días hasta la fecha " + fechaRowActual + " : "+ str(numeroMencionesTotales))
