**skrypt_publ_dane_v5.py**:

Generuje pliki danych dla grid_search:
  data_UT1_v5_048.csv  - dane treningowe z liczebnością rekordów klasy 1 (uszkodzenie 1) 48% klasy 0
  
  data_UT1_v5_025.csv  - dane treningowe z liczebnością rekordów klasy 1 (uszkodzenie 1) 25% klasy 0
  
  data_UT1_v5_010.csv  - dane treningowe z liczebnością rekordów klasy 1 (uszkodzenie 1) 10% klasy 0
  
  data_UT2_v5.csv  -  dane testowe uszkodzenia 2
  
  data_UT3_v5.csv  -  dane testowe uszkodzenia 3

 
**skrypt_publ_grid_samplers_v5.py**

Wykonuje gridsearch na modelu MLP i przygotowanych plikach 
	
**skrypt_publ_wyniki_sampler_v5.py**
Generuje wyniki predykcji uszkodzenia 2 i 3 na modelach stworzonych przez gridsearch
