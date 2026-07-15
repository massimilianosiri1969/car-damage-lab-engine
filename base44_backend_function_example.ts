/*
Esempio concettuale per una backend function Base44/Deno.
La gestione concreta del file può cambiare in base a come Base44 restituisce
il riferimento UploadFile. Non inserire la chiave OpenAI in questa funzione:
qui serve soltanto DAMAGE_API_URL.
*/

Deno.serve(async (req) => {
  try {
    const incoming = await req.formData();

    const image = incoming.get("image");
    const mask = incoming.get("mask");
    const severity = incoming.get("severity_percent");
    const area = incoming.get("area_percent");

    if (!(image instanceof File) || !(mask instanceof File)) {
      return Response.json(
        { error: "Fotografia o maschera mancanti." },
        { status: 400 },
      );
    }

    const apiUrl = Deno.env.get("DAMAGE_API_URL");
    if (!apiUrl) {
      return Response.json(
        { error: "Secret DAMAGE_API_URL non configurato." },
        { status: 500 },
      );
    }

    const outgoing = new FormData();
    outgoing.append("image", image, image.name || "source.jpg");
    outgoing.append("mask", mask, mask.name || "mask.png");
    outgoing.append("severity_percent", String(severity ?? "0"));
    outgoing.append("area_percent", String(area ?? "0"));
    outgoing.append("output_quality", "medium");

    const response = await fetch(`${apiUrl}/v1/damage/edit`, {
      method: "POST",
      body: outgoing,
    });

    const body = await response.text();
    return new Response(body, {
      status: response.status,
      headers: { "content-type": "application/json" },
    });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : "Errore inatteso" },
      { status: 500 },
    );
  }
});
