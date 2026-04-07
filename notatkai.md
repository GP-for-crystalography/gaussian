Co rozwiązujemy za problem?

- `Suprvised learning` czyli mamy dane wejściowe $D = {{x, y}}$ i na ich podstawie szukamy funkcji $f$ która zadziała też na  zbiór testowy $x_* \notin D$

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
można jeszcze wprowadzać funkcje celu! robiliśmy do tej pory bez tego i daliśmy rade znaleźć rozkład