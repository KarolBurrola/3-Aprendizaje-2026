import random
from collections import Counter
import arboles_numericos as an


def entrena_bosque_al(datos, target, clase_default, m_subconjuntos=10, variables_por_nodo=None, **kwargs):

    arbolesb = []

    for j in range(m_subconjuntos):
        subconjuntos = random.choices(datos, k=len(datos))

        arbol = an.entrena_arbol(
            datos=subconjuntos,
            target=target,
            clase_default=clase_default,
            variables_seleccionadas=variables_por_nodo,
            **kwargs
        )
        arbolesb.append(arbol)

    return arbolesb


def predice_instancia_ba(arbolesb, instancia):

    pred = [arbol.predice(instancia) for arbol in arbolesb]
    counter = Counter(pred)
    resp = counter.most_common(1)[0][0]

    return resp


