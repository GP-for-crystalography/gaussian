
# Notatki rozmowa z dominikiem;



nowa metoda zamiast hougha dla EBSD (pomiar całego patternu na ekranie), to ta dominika (ECP pomiar jasności w kole) -> mniejszy koszt obliczeniowy, tanie narzedzia dla mikrospów elektornowych



zmienne to 3 kąty od 0, pi czy coś

identyfikacja konkretnych domen w próbce (można robić)


dane zaszumione wychodzą dominikowi -?> bo mała roóznica w jakiejś długości ścian -> zmienjszyć ilość klas


program, lubi dostać dużo wzórów do generacji na raz


- udostepnic dane pomiarowe

linia nazywa sie linią kikuchiego

## Symulacja:

dominik w słowniku (kąty -> przbieg) ma on 10^6 pozycji



- ile ja bede musiał pozycji żeby dobrze przewidywać

- moze cos zaszumic



## dane dośw:



- 2 materiały coś tam

- dane treningowe ustalić jak je zrobić itp

- porblem w danych doświadczalnych jest dokładnośc do płaszyzny polaryzcji nie do zwrortu

- trzeba uczyć na symulacji która jest legit potwierdzona

- polaryzacje różnia się kolejnoścą ciemne jasne przy prążku na rozecie

- po przemnożeniu skalarnie wyników z danymi (wybieramy najlepsze orientacje) jest drugi słownik, które roróżnia pseudosymetryczne orientacje (różnica miedzy zwrotami w konretnych kierunkach) które zaczynamy rozróżniać

- drugi słownik robi sie te 8h i znowu mnożenie

- dominikowi dobrze już chodzi sam kierunek (bez zwrotu) można na tej podstawie zweryfikować działanie algosa, przed dalszą praca nad wykrywaniem kierunku polryzacji (pseudo symetrycznej orientacji)




# Pomysły do przmeyślenie:


- analiza fourierowska

- przemyśleć czy iloczyn sklaarny szybciej nie policzy P = D E^T (może iloczyn skalarny w fourierze) aktualnie mnożenie macierzy 40min fest ramu

- wygnerowanie macierzy D (wynki symulacji) 10^6 trwa 8h

- przy drugim mnożeniu troche głupieje kod - czemu?

- coś numerycznie lepiej przekmninić

- program lubi batche -> jakoś to obejśc (dawać batche ale mniejsze)

- hough nie działa.... bo nie mamy całej mapy...



## info o programie



- program musi uwzglednic abberacje sferyczna

- omijamy to generujac małe mapki w interesujacych nas punktach



TECHNICZNE DO PRACY:
- upewnić sie że uzywa echord xd
- zrobić grafike przepływu danych
- mozliwe ze trzeba zmienic katy eulera na kwwqterniony lub cosinusy -> chyba ze juz są!
- zrobic na uv

