# Car Damage Lab – Backend sperimentale

Backend FastAPI per una web app che modifica separatamente:

- gravità della deformazione: da -100% a +100%;
- superficie danneggiata: da -100% a +100%;
- incremento dell'interfaccia: 1%.

## Avvio locale

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload
```

Aprire:

- API: `http://localhost:8000`
- documentazione interattiva: `http://localhost:8000/docs`
- controllo: `http://localhost:8000/health`

Il file `.env.example` parte con `MOCK_MODE=true`: il backend restituisce
un'anteprima rossa della superficie interessata, senza usare l'AI.

Per attivare l'editing:

```env
MOCK_MODE=false
OPENAI_API_KEY=...
OPENAI_IMAGE_MODEL=gpt-image-2
```

## Convenzione della maschera

La maschera inviata dalla web app deve avere le stesse dimensioni della foto:

- bianco: area che può essere modificata;
- nero: area da proteggere.

Il backend converte questa maschera nel formato richiesto dal motore immagini.

## Endpoint

`POST /v1/damage/edit` come `multipart/form-data`

Campi:

- `image`: fotografia;
- `mask`: PNG;
- `severity_percent`: intero -100/+100;
- `area_percent`: intero -100/+100;
- `output_quality`: low, medium, high, auto.

La risposta contiene `result_base64`, trasformabile nel browser in:

```javascript
const src = `data:${response.mime_type};base64,${response.result_base64}`;
```

## Limite sperimentale

L'incremento dell'1% è esatto nei controlli e nell'espansione geometrica della
maschera. La variazione generativa della gravità non è ancora una misura fisica
millimetrica e due valori consecutivi possono produrre differenze minime o non
perfettamente monotone. Tutti i test devono essere salvati per calibrare il modello.
