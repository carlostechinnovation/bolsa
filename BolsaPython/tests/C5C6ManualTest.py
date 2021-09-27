import unittest
import sys
import io


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


    if __name__=="__main__":
        print("Entrando por el main...")
        unittest.main()

