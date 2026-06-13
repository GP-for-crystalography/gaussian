# `Statictial modelling vs Machine learining`

no dla nas to nie bedzie to samo (wg. niektórych jest)

różnica polega na tym, że modelowanie statystyczne jest `wytłumaczalne` i opiera się na wyborze najbardzije prawdopodobnego

Trzy różne pytania w modelowaniu

1. Co się stanie dalej? `Predykcja`
2. Dlaczego to się stało? `Wytłumaczenie i zrozumienie`
3. Co powinniśmy zrobić przy danych kosztach i ryzykach? `Decyzja`

Różnica między `generative` i `predictive`

### `predictive`

chcemy znaleźć podsyztą pod procesem funkcje

$$ \hat{y} = f(x)$$

lub jej rozkład prawdopodobieństwa żeby liczyć najczęstrzy przy zadanych warunkach etc.

$$ p(y|x)$$

tutaj. np. ML

### `generative`

chcemy określić jak dane mogą być generowane (produkowane) przy pewnych założeniach o $\theta$

$$ p(y|x, \theta)$$

wtedy model jest **symulatorem**, który generuje sztuczne dane

one są bardzo mocne bo można je `przetestować`:

- ma obserwowalne i przewidywalne wnioski
- interpretowalny
- można je krytykować
- **to nie model to cała historia założeń co się dzieje**

generacyjne modele również umieją przewidywać na podstawie bayesa - ale potrafi być to bardziej kosztowne

bardzo ważne jest aby być w stanie powiedzieć jak pewni czegoś jesteśmy daltego potrzebujemy znać niepewność

`Bayesowskie wnioskowanie`

Założenia Analizy Bayesowskiej:

- dane są stałe, parametry się zmieniają
- prior założenia są w modelu i wnioskowaniu
- wszystko ma rozkład prawdopodobieństwa

Chceym zakutalizować naszą statystykę na wyściu przy pewnych pierwontych założeniach i przy występowaniu danych

- `Prior` $p(\theta)$ -> okreśa co jest możliwe zanim zobaczymy dane, nasze założenia np. częstość występowania słowa w tekstach
- `Likelihood` $p(y|\theta)$ ->jakie dane przewidujemy przy naszych zalożeniach, no wynika z założeń o modelu pod spodem np. pisanie na klawiaturze QWERTY
- `Posterior` $p(\theta|y)$ -> to zaktualizowane prawd. po zobaczeniu danych, bierze wszystko pod uwage

$$ p(\theta|y) = \frac{p(y|\theta) \ p(\theta)}{\sum\_{\theta_i} p(y|\theta_i) \ p(\theta_i) } = \frac{p(y|\theta) \ p(\theta)}{p(y)}$$

stąd w praktyce jako, że mianownik nie zależy od $\theta$ najważniejsze jest

$$p(\theta|y) \propto p(y|\theta) \ p(\theta)$$
