import os
import random
import utileria as ut
import arboles_numericos as an
import bosque_aleatorio as ba


urldatos = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
directorio_datos = "datos"
rutazip = os.path.join(directorio_datos, "cancer.zip")
rutadatos = os.path.join(directorio_datos, "wdbc.data")

if not os.path.exists(directorio_datos):
    os.makedirs(directorio_datos)

if not os.path.exists(rutazip):
    ut.descarga_datos(urldatos, rutazip)
    ut.descomprime_zip(rutazip)

nombres = ['ID', 'Diagnosis'] + [f'caracteristica_{i}' for i in range(1, 31)]
datos = ut.lee_csv(rutadatos, atributos=nombres)

for registro in datos:
    registro['Diagnosis'] = 1 if registro['Diagnosis'] == 'M' else 0

    for i in range(1, 31):
        registro[f'caracteristica_{i}'] = float(registro[f'caracteristica_{i}'])
    del (registro['ID'])

target = 'Diagnosis'

random.seed(70)
random.shuffle(datos)

m = int(0.8 * len(datos))
dentrenamiento = datos[:m]
dvalidacion = datos[m:]

print("Prueba 1: Efecto del número de árboles")

arboln = an.entrena_arbol(dentrenamiento, target, clase_default=0, max_profundidad=5)
exactitudarb = an.evalua_arbol(arboln, dvalidacion, target)
print(f"Exactitud del árbol: {exactitudarb:.4f}")

lista = [20,60,100]
for num_arb in lista:
    bosque_actual = ba.entrena_bosque_al(
        dentrenamiento,
        target,
        clase_default=0,
        m_subconjuntos=num_arb,
        variables_por_nodo=5,
        max_profundidad=5
    )
    exactitudbos = ba.evalua_bosque_al(bosque_actual, dvalidacion, target)
    print(f"Bosque aleatorio con {num_arb:3} árboles | Exactitud: {exactitudbos:.4f}")
