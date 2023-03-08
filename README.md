# MSI-self-supervised-clustering


### For DEVS
___
Dane do przetestowania algrytmów:  
https://drive.google.com/drive/u/1/folders/1C04FcG6QzxF6dE5n4qTk0ArI1-b8Cn3H  
(plik example_results_and_dane.zip, rozpakować i katalogi /dane oraz /resuts dodać bezpośrednio do źródłowego katalogu projektu)
___
__Workflow:__  
1. Dodać dane do katalogu dane.
2. Doprowadzić dane do takiej formy jaką można otrzymać w wyniki uruchomienia **bad_np_to_good_(ver).py**,
jeśli dane są w formacie imzml, to być może **imzml_to_numpy.py** pomoże.
3. Uruchomić **np_to_parquet.py** (w przyszłości ten krok można ominąć i przpisać kod na niekorzystanie z parquet), zapisuje on dane w innej strukturze.
4. Obliczenia:  
    a. uruchomić **original_kmeans.py**, aby policzyć klastry na spektrach.  
    b. uruchomić trening sieci**.  
    bb. uruchomić contrastive_kmeans.py, aby policzyć klastry na reprezentacjach spektr.  
5. uruchomić evaluate.py aby policzyć wyniki, narysować obrazki.  

** kod do treningu sieci jest na google colab: https://drive.google.com/drive/u/1/folders/1C04FcG6QzxF6dE5n4qTk0ArI1-b8Cn3H
___

### Future Work:
Zalążki XAI w **evaluate_lime.py**.
