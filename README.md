# MsFEM 1D — Méthode des éléments finis multi-échelles

Implémentation pédagogique en Python de la **Multiscale Finite Element Method
(MsFEM)** pour un problème elliptique 1D à coefficient fortement oscillant,
comparée à la méthode P1 classique.

Projet EDP / éléments finis — M2 MACS.

## Problème modèle

On résout sur $(0,1)$, avec conditions de Dirichlet homogènes :

$$-\frac{d}{dx}\!\left(A(x)\,\frac{du}{dx}\right) = f, \qquad u(0)=u(1)=0,$$

avec un coefficient oscillant à petite échelle $\varepsilon$ et un second membre constant :

$$A(x) = \frac{1}{2 + \cos\!\left(\dfrac{2\pi x}{\varepsilon}\right)}, \qquad f \equiv 1.$$

Ce problème possède une **solution exacte en forme close** (voir
[`msfem_1d/problem.py`](msfem_1d/problem.py)), ce qui permet de mesurer les
erreurs sans quadrature de référence sur une solution numérique.

- Coefficient homogénéisé : $A_\text{hom} = 1/2$ (moyenne harmonique).
- Solution homogénéisée : $u_0(x) = x(1-x)$.

### Enjeu

La P1 classique n'est précise que si le pas $H \ll \varepsilon$ (elle « voit »
les oscillations). Dans le régime résonant $H \approx \varepsilon$, elle
sous-estime la solution d'un facteur $\sqrt{3}/2$. La **MsFEM** capture les
oscillations via des **fonctions de base construites en résolvant des problèmes
locaux** sur un sous-maillage fin, et reste précise avec un maillage grossier.

## Idée de la méthode

Sur chaque maille grossière (pas $H = 1/N$), les fonctions de base MsFEM
$\Phi_i$ remplacent les chapeaux P1. Elles résolvent le problème homogène local
$-(A\,\Phi_i')' = 0$ avec les valeurs nodales de la P1 comme conditions au bord.
Elles s'adaptent donc à $A(x)$ sur chaque cellule (sous-maillage fin de pas
$h = H/n$). Le problème global est ensuite assemblé et résolu sur ces bases.

## Structure du projet

```
MsFEM/
├── main.py                 # Études : comparaisons, convergence, benchmark, test résonant
├── plot_exact.py           # Trace la solution exacte pour plusieurs ε
├── scripts/
│   └── plot_A.py           # Trace le coefficient oscillant A(x)
├── msfem_1d/               # Package principal
│   ├── problem.py          # A, f, solution exacte close, solution homogénéisée
│   ├── mesh.py             # Mesh1D : maillage grossier + sous-maillage fin
│   ├── fem_p1.py           # Éléments finis P1 classiques
│   ├── msfem.py            # Solveur MsFEM : bases locales + assemblage global
│   ├── solution.py         # DiscreteSolution / P1Interpolant (interface value/grad)
│   ├── errors.py           # Normes L2, H1, H1-semi, énergie
│   ├── plots.py            # Tracés : solutions, erreur ponctuelle, convergence
│   └── timing.py           # Chronométrage des solveurs
├── tests/                  # Tests pytest (unitaires, convergence, solution exacte)
├── assemblage_pb_global.pdf   # Schéma d'assemblage du problème global
└── assemblage_pb_local.pdf    # Schéma d'assemblage des problèmes locaux
```

## Installation

Nécessite Python 3 avec **NumPy** et **Matplotlib** (**pytest** pour les tests).

```bash
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install numpy matplotlib pytest
```

## Utilisation

```bash
python main.py            # comparaisons, convergence, benchmark et test résonant
python plot_exact.py      # solution exacte pour plusieurs ε
python scripts/plot_A.py  # coefficient oscillant A(x)
```

`main.py` enchaîne cinq études et sauvegarde les figures en PNG :

1. **Comparaison visuelle** — P1 fin vs P1 grossier vs MsFEM vs solution exacte.
2. **Fonctions de base** — bases MsFEM $\Phi_i$ vs chapeaux P1.
3. **Convergence en $H$** — erreurs L2 / H1 de la MsFEM et de la P1 grossière.
4. **Benchmark** — temps de résolution et diagramme travail-précision.
5. **Test résonant** — vérification du facteur $\sqrt{3}/2$ à $H = \varepsilon$.

### Exemple d'API

```python
from msfem_1d import Mesh1D, fem_p1_solve, msfem_solve, exact_solution
from msfem_1d.problem import A, f

u_ms    = msfem_solve(Mesh1D(N=8, n=32), A, f)   # MsFEM : 8 mailles grossières, 32 fines/maille
u_p1    = fem_p1_solve(Mesh1D(N=8, n=1), A, f)   # P1 grossière
u_exact = exact_solution(eps=1/16)

val = u_ms.value(0.5)        # évaluation ponctuelle
```

## Tests

```bash
pytest
```

Les tests couvrent la forme close de la solution exacte
([`tests/test_exact.py`](tests/test_exact.py)), les briques élémentaires du
solveur MsFEM — quadrature de Simpson, conditions au bord et flux des problèmes
locaux, réduction au chapeau P1 quand $n=1$
([`tests/test_msfem_unit.py`](tests/test_msfem_unit.py)) — et le comportement
de convergence, dont la supériorité de la MsFEM sur la P1 grossière en régime
résonant ([`tests/test_msfem_convergence.py`](tests/test_msfem_convergence.py)).

## Notes

Les figures générées (`*.png`) sont ignorées par Git (voir
[`.gitignore`](.gitignore)).
