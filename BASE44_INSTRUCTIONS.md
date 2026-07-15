# Istruzioni da incollare nella chat di Base44

Crea una web app desktop-first e responsive chiamata **Car Damage Lab**.
È uno strumento sperimentale separato da CarPractice. Non aggiungere login,
pagamenti, menu complessi o funzioni diverse da quelle descritte.

## Obiettivo

L'utente carica o scatta la fotografia di un'automobile danneggiata, seleziona
la zona del danno, imposta due percentuali e genera una fotografia modificata:

1. **Gravità della deformazione**, da -100% a +100%.
2. **Superficie danneggiata**, da -100% a +100%.

Entrambi i valori devono essere numeri interi e modificabili esattamente
dell'1% alla volta.

## Layout

Crea una pagina unica con:

- header sottile con logo testuale “Car Damage Lab” e badge “PROTOTIPO”;
- area principale a due colonne su desktop;
- colonna sinistra: fotografia originale e strumenti di selezione;
- colonna destra: risultato generato;
- pannello controlli sotto le immagini;
- su smartphone impila tutti gli elementi verticalmente.

Usa un aspetto tecnico, pulito e professionale automotive:
sfondo grigio chiarissimo, card bianche, testo antracite, accento arancione.
Evita gradienti vistosi, effetti futuristici, animazioni veloci e decorazioni inutili.

## Caricamento foto

Inserisci:

- pulsante “Carica fotografia”;
- pulsante “Scatta fotografia” su dispositivi compatibili;
- formati JPG, JPEG, PNG e WEBP;
- limite iniziale 20 MB;
- anteprima della foto;
- pulsante “Sostituisci foto”;
- messaggi chiari per formato o dimensione non validi.

Usa l'integrazione Base44 `UploadFile` per il caricamento.

## Selezione della zona

Sopra la fotografia originale aggiungi un canvas trasparente con:

- pennello circolare;
- cursore dimensione pennello da 5 a 100 px;
- modalità “Seleziona danno”;
- modalità “Cancella selezione”;
- pulsante “Annulla ultimo tratto”;
- pulsante “Cancella selezione”;
- sovrapposizione semitrasparente arancione sulla zona selezionata.

Quando l'utente genera, esporta una maschera PNG delle stesse dimensioni
della fotografia:

- pixel bianchi nella zona selezionata;
- pixel neri in tutto il resto.

Non consentire la generazione se non è stata selezionata una zona.

## Controllo gravità

Crea una card chiamata “Gravità della deformazione” con:

- pulsante meno;
- campo numerico centrale;
- simbolo percentuale;
- pulsante più;
- slider da -100 a +100;
- step obbligatorio 1;
- valore iniziale 0;
- pulsante “Reimposta a 0”.

Ogni clic su meno o più modifica il valore esattamente di un punto.
Il campo accetta solo numeri interi compresi tra -100 e +100.
Le frecce della tastiera devono modificare il valore di 1.
Non permettere valori esterni all'intervallo.

Mostra sotto:
“Valori negativi riducono o riparano il danno. Valori positivi lo aggravano.”

## Controllo superficie

Crea una card identica chiamata “Superficie danneggiata”, con:

- intervallo -100/+100;
- step 1;
- valore iniziale 0;
- pulsanti meno e più;
- campo numerico;
- slider;
- reimpostazione a 0.

Mostra sotto:
“Valori negativi restringono la zona. Valori positivi la estendono.”

## Generazione

Aggiungi il pulsante principale “Genera modifica”.

Al clic:

1. verifica che foto e maschera siano presenti;
2. mostra stato “Elaborazione in corso”;
3. disabilita il pulsante per evitare richieste duplicate;
4. invia fotografia, maschera e valori al motore esterno;
5. mostra la fotografia restituita;
6. in caso di errore mostra un messaggio leggibile e il pulsante “Riprova”;
7. conserva sempre la fotografia originale.

Invia una richiesta multipart/form-data a:

`POST {{DAMAGE_API_URL}}/v1/damage/edit`

Campi esatti:

- `image`: file originale;
- `mask`: maschera PNG;
- `severity_percent`: intero;
- `area_percent`: intero;
- `output_quality`: stringa `medium`.

La risposta JSON sarà:

```json
{
  "job_id": "uuid",
  "status": "completed",
  "mode": "mock oppure ai",
  "severity_percent": 25,
  "area_percent": -10,
  "result_base64": "...",
  "mime_type": "image/jpeg"
}
```

Visualizza il risultato con:

```javascript
const resultUrl =
  `data:${response.mime_type};base64,${response.result_base64}`;
```

Non inserire chiavi OpenAI nel frontend.

## Collegamento sicuro consigliato

Crea una backend function Base44 chiamata `editVehicleDamage`.
La funzione deve ricevere file/URL e parametri dalla pagina e inoltrarli al
motore esterno. Salva `DAMAGE_API_URL` come Secret Base44, non nel codice
pubblico. La funzione deve restituire integralmente il JSON ricevuto dal motore.

In alternativa, importa `openapi.json` come Custom Integration del workspace
e sostituisci nel file il server provvisorio con l'URL reale del backend.

## Confronto

Dopo la generazione mostra tre modalità:

- “Originale”;
- “Risultato”;
- “Confronta”.

In “Confronta”, usa uno slider verticale prima/dopo trascinabile.
Aggiungi pulsanti:

- “Scarica risultato”;
- “Nuova prova con gli stessi valori”;
- “Ripristina valori”;
- “Nuova fotografia”.

## Registro prove

Crea l'entità `DamageTest` con:

- `original_file_url`;
- `mask_file_url`;
- `result_data_url` o `result_file_url`;
- `severity_percent`;
- `area_percent`;
- `job_id`;
- `engine_mode`;
- `status`;
- `error_message`;
- `created_date`.

Mostra sotto la pagina le ultime 10 prove come miniature, senza creare
una pagina gestionale separata.

## Regole importanti

- I valori devono restare indipendenti.
- Non generare automaticamente a ogni variazione dell'1%.
- Genera solo premendo il pulsante.
- Non applicare filtri alla foto nel browser.
- Non ridimensionare o ritagliare visivamente la fotografia senza mantenere
  le coordinate corrette del canvas.
- La maschera deve corrispondere pixel per pixel alla foto originale.
- Mostra sempre i valori usati accanto al risultato.
- Inserisci il disclaimer:
  “Simulazione AI sperimentale. Le percentuali indicano una variazione visiva
  e non una misurazione fisica certificata.”
