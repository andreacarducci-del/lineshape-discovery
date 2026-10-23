# lineshape-discovery

Progetto di ricerca: un LLM propone parametrizzazioni di ampiezze per lineshape
adroniche vicino a soglia, un verifier le fitta e ne controlla la consistenza
fisica, e un critico stima il valore di una proposta *prima* del fit.
Domanda centrale: quanto bene si può predire il valore di un'ipotesi prima di
eseguirla, rispetto a un LLM-judge e a baseline senza critico.

L'autore è un dottorando in fisica adronica teorica: conosce bene la teoria
dello scattering, è meno esperto di ingegneria ML. Il progetto deve dimostrare
che sa far girare esperimenti ML riproducibili, quindi deve capire ogni riga.

## Ambiente

- Windows, PowerShell, Python 3.12 in `.venv` (attivare con `.\.venv\Scripts\Activate.ps1`).
- Test: `pytest` dalla cartella principale (`pytest.ini` imposta `pythonpath = .`).
- Figure: `python -m experiments.<nome_script>`, output in `experiments/figs/`.

## Struttura

- `src/` codice di libreria (`amplitudes.py`: fase spazio, Breit–Wigner, Flatté).
- `tests/` test pytest; ogni funzione fisica ha almeno un test.
- `experiments/` script che producono figure e risultati.
- `configs/` configurazioni degli esperimenti; `notes/` diario e note di lettura.

## Convenzioni fisiche (non cambiarle senza chiedere)

- Variabile s in GeV², complessa in generale; masse in GeV.
- Fase spazio rho(s) = 2 q(s) / sqrt(s).
- Normalizzazione S = 1 + 2 i rho f, quindi unitarietà elastica: Im(1/f) = -rho.
- Foglio I: Im q >= 0; foglio II: q -> -q. Con due canali i fogli sono
  etichettati dai segni di (Im q1, Im q2).
- Natura dei poli: legato = reale sotto soglia sul foglio I; virtuale = reale
  sotto soglia sul foglio II; risonanza = complesso sul foglio II.

## Regole di lavoro

- Prima di scrivere codice, spiega in breve cosa farai e perché.
- Ogni nuova funzione fisica arriva con un test (unitarietà, limiti noti, poli noti).
- Esegui `pytest` prima di proporre un commit; non committare con test rotti.
- Seed fissati ovunque ci sia casualità; niente risultati che non si riproducano da un comando.
- Codice semplice e leggibile, niente astrazioni premature.
- Messaggi di commit in italiano, brevi.

## Stato

Piano di 12 settimane (fine ottobre 2026 – fine gennaio 2027).
Fatto: setup del repo, ampiezze Breit–Wigner e Flatté a due canali con test di unitarietà.
Prossimo: generatore di dati sintetici (`make_data.py`), poi fit con iminuit e ricerca dei poli.
