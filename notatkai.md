słowniczek

`inference` = wnioskowanie

`latent` = ukryty

Co rozwiązujemy za problem?

- `Suprvised learning` czyli mamy dane wejściowe $D = {{x, y}}$ i na ich podstawie szukamy funkcji $f$ która zadziała też na zbiór testowy $x_* \notin D$

Idea procesów gaussowskich

- w przeciwnieństwie do rozwiązań `parametrycznych` gdzie postać $f$ jest zakładana i szukane są parametry (np. funckja liniowa), my nie zakładamy postaci f, tylko myślimy poziom wyżej na poziomie przestrzeni funkcji, którą sami wybraliśmy, i na niej narzucamy rozkład prawdopodobieństwa wynikające z $D$

# `weighted space`

jest to jedna z 2 perspektyw patrzenia na procesy gausowskie

potem jest przykład wyprowadzenie regresji liniowej dla R^n -> R z perspektywy Bayesowskiej, czyli szukamy funkcji liniowej, która w obecności szumy bedzie dawała najmenijszy bład!

potemm w schemacie bayesowskim wyliczamy Rozkład z tw. bayesa, który pozwala nam określic $p(w|y, X)$ czyli prawdopodobieństwo wag w regresji przy zadanym D = {X, y}, potem nie określa się kryterium jak w parametrycznych tylko liczymy funkcję
$f_∗ =  f(x_∗)  x∗$ uśredniając po wszystkich możliwych modelach w tym przypadku po wszystkich wagach modeli
$p(f_*, x_*, X, y) = \int p(f_*| x*, w) p(w|X, y) dw$ i dostajemy $f_*$

potem to samo można określić dla dowolnej przestrzeni funkcji nie koniecznie $R^n$ robią sobie z nią izomorfizm $\phi(x)$ między bazami w danych przestrzeniach i w zasadzie dostaje sie te same wzory co w wersji liniowej

potem jest pokazany trick z przekrztałceniem algebraicznym na problem odwracania innej macierzy -> co oznacza, że przestajemy szukać w przestrzeni w której to naprawde liczymy (feature space), tylko szukamy w tym $R^n$ do którego sprowadzamy problem to sprawia, że jest mniej obliczeń gdyż $n < N$

tak samo widzimy, że w kółko pojawia sie liczenie, takiego wyrażenia jak $\phi(x)^T
\Sigma_p \phi(x')$, które w zasadzie jest indukowanym iloczynem skalarnym w $R^n$... co oni nazywają `kernel` ale ja bym nazwał `metryką` ;pp

# podejście `function space`

definicja procesu gaussowskiego ($GP$)

- to zbiór indeksownay (np. czasem) zbiór zmiennych losowych ${x_i}$ -> $x_i$ to zmienna losowa!
- których dowolna skończona kombinacja ma rozkład normalny
- potem dla dowolnego rzeczywistego procesu $f(x)$ deiniujemy średnią i funkcje kowariancji, co calkowicie determinuje $GP$

$m(x) = 𝔼[ f(x)]$

$k(x, x') = 𝔼[(f(x) - m(x))(f(x') - m(x'))]$

w tej perspektywie data set treningowy oznaczmy jako $(x_i, f_i) = (x_i, f(x_i)) = (x_i, y_i)$ f_i to też zmienna losowa

ma to coś co sie nazywa marginalization property (consistency) -> czyli podzbiór nie przeczy całości

jak wcześniej oznaczaliśmy mając zbiór testowy $x_*$, mamy też dla niego do obliczenie wartość $f_*$ funkcji dla danych testowych. Zabawa polega na tym żeby znaleźć rozkład prawdopodobieństwa w przestrzeni funkcji który jak najelepiej odda $f_*$, w tym celu postuluje się jakąś funkcje kowariancji, gdzie najcześciej wybiera się

$cov(f(x_p, x_q)) = k(x_p, x_q) = exp(-\frac{1}{2}||x_p-x_q||^2)$

później bierze się zmienna losową $X_*$ i z niej losujemy jakieś punkty, liczymy na podstawie ich macierz kowariancji i generujemy Guassa następującego

$f_* \approx N(0, K(X_*, X_*)$

stąd `istotne jest wiedzieć jak sie generuje wielowymiarowego Gaussa`

`w przypadku dokładnych danych`
można jeszcze podbić stawke i robić rozkład na podstawie zarówno $X$ jak i $X_*$ i wzięciu prawdopodobieństwa warunkowego

autor odróżnia $x$ (pojedyczny sample) i $X$ (jakiś ciąg sampli)

`gdy mamy zaszumione dane wyjściowe`
to robimy w zasadzie to samo i dostajemy te same wnioski co w wersji `weighted space` z szumem na wyjściu...

no i okazuje sie że obliczenie rozkładu do wyniku sprowadza się do odwrócenia symetrycznej macierz kowariancji może z dodanym szumem (Cholesky + basksub) ale dalej w tym nieparametrycznym modelu mamy parametry XD nazywane `HIPERPARAMETRAMI` `kewk serio... które determinują jaką funkcje wariancji weżmiemy i jaki szum założymy

# ogarnianie hiperparametrów w regresji

można jeszcze wprowadzać funkcje celu! robiliśmy do tej pory bez tego i daliśmy rade znaleźć rozkład.

funckcja kary = loss function (oni oznaczają $\mathcal{L}(y_{true}, y_{guess})$)

jest jakieś zdanie `The likelihood and loss function need not have anything in common.` i nie daj sb wmówić że jest inaczej

ok to, jak działa kara? no chcemy znaleźć wartość $y_{guess}$ odbiega od $y_{true}$, no ale nie znamy go XD, zatem bierzemy $y_{guess}$, który minimializuje oczekiwaną karę w naszym bayesowskim modelu, czyli Risk

$\tilde{R}_\mathcal{L}(y_{guess}|x∗) = \int\mathcal{L}(y∗,y_{guess})p(y∗|x∗, D) dy∗$

tak sie składa że tym optymalnym dla $\mathcal{L}$

- $|x_1 - x_2|$ jest mediana
- $(x_1 - x_2)^2$ jest średnia

dla gaussowskiej sa one równe!

jest potem przykład z kinemematyka odwrotną robota, gdzie GPR (jakiś reduced) linear regressin wypada najlepiej co do przewidywań. Problemem jest, kostowność $O(n^3)$

potem jest opis tego że GP linear reggression jest czymś co nazywa się `linear smoothener`, gdzie predykcja jest kombinacją liniową treningowych danych, gdzie regresja liniowa jest kombinacją liniową tylko inputów

jest jeszcze opcja przybliżania liczenia tego, np. jak `kernel function`

# `klasyfikacja`

ponownie chcemy znaleźć funkcje aproksymująca ta prawdziwą pod spodem, jednak teraz przestrzeń wyjść jest dyksretnym zbiorem klas $\mathcal{Ci}$, z tego powodu rozkład Gaussowski, który używaliśmy w przypadku regresji się nie nadaje...

bedziemy generalizować regresje logistyczną...

są niby dwa podejścia do obliczania szansy przypisania do klasy $p(y|\mathbf{x})$

- generative (wzór Bayesa i liczysz posterior prob na podstawie prior)
- disciminative (liczymy bezpośrednio rozkład prawdopodobieństwa przypisania do klasy)

discimintive robi się najprosciej sprowadzając problem do regresji, w przypadku binarnej robi sie to funkcją logistyczną

$λ(z) = \frac{1}{1 + exp(−z)}$

można używać innych funkcji jak np. dystrubuanty rozkładu normalnego to sie nazywa probit

my się zajmiemy discrimative

podejmowanie decyzji, do tego wykorzytujemy funkcje kary, kosztu, loss fucniotn $\mathcal{L}(c, c')$

prostą funkcją kary jest zero-jedynkowa, czyli 1 gdy źle 0 gdy dobrze zaklasyfikujemy, nie zawsze dobre (nie widzi false positeve kosztu) optymalny 01 klasyfikator nazywa sie `Bayes classifier`

my robimy probabilistic approach gdzie wybieramy tak by minimalizować ryzoko prawdopodobieństw przy decyzji z Bayesa, można by też szukać rozkładu decyzji $c(x)$, który minimalizuje ryzyko całkowite całką liczone, ale to jest trudne

no i sprowadza sie to do wyliczenia wektora wag  a potem odwrócenia tej nieliniowej zmiany baz, którą zrobiliśmy, mamy to szczęście, że wsyzstko jest wypukłe i można wyliczać newtonem...

`przybliżenie laplaca` polega na pzybliżeniu funkcji logit lub innej za pomocą gaussa z aproksymacji drugiego rzędu

- najpierw szukamy maksimum prawdziwego rozkładu $\hat{f}$ metodą newtona (bo funkcja jest wypukła)
- potem wyliczamy odpwoeidnie pochodne i rpzybliżamy Gaussem

alternatywa to `expectation propagation`
- tu sie przybliża jakimiś gaussami poszczególne rozkłady punktów
- jakiś iteracyjnyt algorytm 
  
porównanie metod:
- laplace pomija nie kładze nacisku na punktu "explainable" czyli takie które łatwo wydedukować
- 