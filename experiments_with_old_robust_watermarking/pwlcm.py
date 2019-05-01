# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-
import math


def mypwlcm(dic, valores):
    '''
    dic: diccionario compuesto por la semila y el valor de p
    valores: lista de valores a tratar
    '''
    valores_finales = []
    cantidad_valores = len(valores)
    posiciones = orden(dic, cantidad_valores)
    # print 'Posiciones: ', posiciones
    posiciones_distintas = lista_valores_distintos(posiciones)
    # print 'Posiciones distintas: ', posiciones_distintas
    if len(posiciones_distintas) == len(posiciones):
        v = []
        for i in range(len(posiciones_distintas)):
            v.append(valores[i])
        return v
    if len(posiciones_distintas) == 1:
        return valores
    for i in range(len(posiciones_distintas)):
        valores_finales.append(valores[posiciones_distintas[i]])
    posiciones_faltantes = lista_valores_faltantes(
        posiciones_distintas, cantidad_valores)
    # print 'Faltantes: ', posiciones_faltantes
    if len(posiciones_faltantes) > 0:
        v = []
        for i in range(len(posiciones_faltantes)):
            v.append(valores[posiciones_faltantes[i]])
        valores_finales.extend(mypwlcm(dic, v))
    return valores_finales


def pwlcm(dic):
    # dic: diccionario compuesto por la semila y el valor de p
    if (dic['semilla'] >= 0) and (dic['semilla'] < dic['p']):
        x = dic['semilla'] / dic['p']
    elif (dic['semilla'] >= dic['p']) and (dic['semilla'] < 0.5):
        x = (dic['semilla'] - dic['p']) / (0.5 - dic['p'])
    elif (dic['semilla'] >= 0.5) and (dic['semilla'] < 1):
        dic['semilla'] = 1 - dic['semilla']
        x = pwlcm(dic)
    return x


def orden(dic, cant=40):
    lista = []
    for i in range(cant):
        if i != 0:
            dic_a = {}
            dic_a['semilla'] = temp
            dic_a['p'] = dic['p']
            temp = pwlcm(dic_a)
        else:
            temp = pwlcm(dic)
        lista.append(int(math.floor(math.fmod(temp * 10**14, cant))))
    return lista


def lista_valores_distintos(lista):
    lista_nueva = []
    for i in lista:
        if i not in lista_nueva:
            lista_nueva.append(i)
    return lista_nueva


def lista_valores_faltantes(posiciones_distintas, cantidad_valores):
    posiciones_faltantes = []
    for i in range(cantidad_valores):
        if i not in posiciones_distintas:
            posiciones_faltantes.append(i)
    return posiciones_faltantes


def mypwlcm_limit(dic, valores, limite):
    '''
    dic: diccionario compuesto por la semila y el valor de p
    valores: lista de valores a tratar
    '''
    valores_finales = []
    cantidad_valores = len(valores)
    posiciones = orden(dic, cantidad_valores)
    # print 'Posiciones: ', posiciones
    posiciones_distintas = lista_valores_distintos(posiciones)
    # print 'Posiciones distintas: ', posiciones_distintas
    if len(valores_finales) > limite:
        return valores_finales
    if len(posiciones_distintas) == len(posiciones):
        v = []
        for i in range(len(posiciones_distintas)):
            v.append(valores[i])
        return v
    if len(posiciones_distintas) == 1:
        return valores
    for i in range(len(posiciones_distintas)):
        valores_finales.append(valores[posiciones_distintas[i]])
    posiciones_faltantes = lista_valores_faltantes(
        posiciones_distintas, cantidad_valores)
    # print 'Faltantes: ', posiciones_faltantes
    if len(posiciones_faltantes) > 0:
        v = []
        for i in range(len(posiciones_faltantes)):
            v.append(valores[posiciones_faltantes[i]])
        valores_finales.extend(mypwlcm_limit(dic, v, limite))
    return valores_finales
