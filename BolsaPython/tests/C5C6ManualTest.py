import unittest
import numpy as np


class C5C6ManualTest(unittest.TestCase):

    def test_isupper(self):  # testing for isupper
        self.assertTrue('HELLO'.isupper())
        self.assertFalse('HELlO'.isupper())

    def test_split(self):  # testing for split
        self.assertEqual('Hello World'.split(), ['Hello', 'World'])
        with self.assertRaises(TypeError):
            'Hello World'.split(2)

    #def test_probarParametros(self):
        #    from bolsa import C5C6Manual  #Lo importa y lo EJECUTA
        #print("probarParametros")
        #debenSumar1 = C5C6Manual.fraccion_train + C5C6Manual.fraccion_test + C5C6Manual.fraccion_valid
        #self.assertEqual(debenSumar1, 1, "las fracciones deben sumar uno")

    #def test_comprobarPrecisionManualmente(self):
        #from bolsa import C5C6Manual  # Lo importa y lo EJECUTA
        #targetsNdArray1 = np.array([[1, 2, 3], [4, 5, 6]])
        #targetsNdArray2 = np.array([[1, 2, 3], [4, 5, 6]])
        #etiqueta = "TEST"
        #id_subgrupo = 7
        #out = C5C6Manual.comprobarPrecisionManualmente(targetsNdArray1, targetsNdArray2, etiqueta, id_subgrupo)
        #self.assertTrue('INTERESANTE' in out)

    if __name__=="__main__":
        print("Entrando por el main...")
        unittest.main()

